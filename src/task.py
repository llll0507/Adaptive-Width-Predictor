import logging
import random
import re
import time
import numpy as np

from src.base import get_answer
from src.mcts import MCTS_search
from utils.inference_model import get_response, get_subq_and_answer_response, get_value_response
from utils.prompts import star_value_prompt
from utils.value_function import risk_value, similarity_value
from utils.wrap import answer_wrap, decompose_wrap, final_answer_wrap


class MCTSTask:
    def __init__(
        self,
        time_limit=None,  # Time limit in milliseconds
        iteration_limit=None,  # Maximum number of iterations
        exploration_constant=1.0,  # UCT exploration constant
        data=None,  # test data
        data_path=None,
        data_idx=None,
        alpha=0.1,  # Value update rate
        ans_weight=0.75,  # knwledge weight
        multihops=6,  # sub query
        total_depth=5,  # Total depth of the tree
        temperature=0.7,  # Sampling temperature
        max_tokens=2048,  # Max tokens for generation
        seed=170,  # Random seed
        max_length=2048,  # Max sequence length
        truncation=True,  # Whether to truncate
        do_sample=True,  # Whether to sample
        max_new_tokens=1024,  # Max new tokens to generate
        low=0,  # Minimum value
        high=1,  # Maximum value
        run_mode="MCTS",
        value_mode="risk",
        value_model="Qwen2.5-7B-Instruct",
    ):
        # Task parameters
        self.run_mode = run_mode
        self.value_mode = value_mode
        self.value_model = value_model

        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant

        # Model parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens

        # State
        self.data = data
        self.data_path = data_path
        self.data_idx = data_idx
        self.node_count = 0
        self.limit_type = time_limit
        self.root_node = None
        self.leaf_nodes = []

        # Value range
        self.low_value = low
        self.high_value = high
        self.alpha = alpha
        self.ans_weight = ans_weight
        self.multihops = multihops
        self.total_depth = total_depth

    def set_limit(self):
        """Set and validate the search limit type (time or iterations).

        Raises:
            ValueError: If both time and iteration limits are set, or if neither is set,
                      or if iteration limit is less than 1-200.
        """
        if self.time_limit is not None and self.iteration_limit is not None:
            raise ValueError("Cannot have both a time limit and an iteration limit")

        if self.time_limit is None and self.iteration_limit is None:
            raise ValueError("Must have either a time limit or an iteration limit")

        if self.time_limit is not None:
            self.limit_type = "time"
        else:
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = "iterations"

    def sub_queries_to_nodes(self, state, num_children):
        """
        Generate sub-queries from the current node with robust error handling.

        Args:
            state: Current reasoning state
            num_children: Number of sub-queries to generate

        Returns:
            list: List of dictionaries containing sub-queries and their answers
        """
        # ========== Step 1: Generate decomposition prompt ==========
        try:
            decompose_prompt = decompose_wrap(state, self.data_path, self.data_idx)
            logging.debug(f"Generated decompose prompt (length: {len(decompose_prompt)})")
        except Exception as e:
            logging.error(f"Error generating decompose prompt: {str(e)}")
            return []

        # ========== Step 2: Get sub-queries (with retry) ==========
        max_retries = 3
        response = []

        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Requesting sub-queries (attempt {attempt}/{max_retries})")

                # Use specialized sub-question decomposition and answering function (may be online model)
                response = get_subq_and_answer_response(
                    decompose_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    do_sample=self.do_sample,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=num_children,
                )

                # ===== Key fix: Add detailed logging =====
                logging.info(f"Response type: {type(response)}")
                logging.info(f"Response length: {len(response) if response else 0}")
                if response:
                    logging.info(f"First element type: {type(response[0])}")
                    logging.info(f"First element: {response[0]}")

                # Validate response effectiveness
                if not response:
                    logging.warning(f"Attempt {attempt}: get_response returned None/empty")
                    response = []
                elif isinstance(response, list):
                    # ===== Key fix: Check both content and generated_text fields =====
                    valid_responses = []
                    for r in response:
                        if isinstance(r, dict):
                            # Priority check generated_text field
                            generated_text = r.get("generated_text", "")
                            if isinstance(generated_text, list) and len(generated_text) > 0:
                                # If generated_text is list, take first element's content
                                content = generated_text[0].get("content", "").strip()
                            elif isinstance(generated_text, str):
                                # If generated_text is string
                                content = generated_text.strip()
                            else:
                                # Fallback check content field
                                content = r.get("content", "").strip()
                            if content:
                                valid_responses.append(r)
                        elif isinstance(r, str) and r.strip():
                            valid_responses.append(r)

                    if not valid_responses:
                        logging.warning(f"Attempt {attempt}: All responses are empty strings")
                        response = []
                    else:
                        response = valid_responses
                        logging.info(f"Attempt {attempt}: Successfully got {len(response)} valid responses")
                        break  # Successfully got valid response, break loop

              
                if not response and attempt < max_retries:
                    wait_time = 2 ** attempt
                    logging.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

            except Exception as e:
                logging.error(f"Attempt {attempt} failed with error: {str(e)}", exc_info=True)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        
        if not response:
            logging.error("Failed to get sub-queries after all retries")
            return []

        # ========== Step 3: Parse sub-queries ==========
        proposed_sub_queries = []
        proposals = []

        try:
            # ===== Key fix: Uniformly handle multiple response formats =====
            if isinstance(response, list) and response:
                for idx, element in enumerate(response):
                    proposal = ""

                    if isinstance(element, dict):
                        # Format 1: {"generated_text": [...]}
                        generated_text = element.get("generated_text", "")
                        if isinstance(generated_text, list) and len(generated_text) > 0:
                            # If list, take first element's content
                            proposal = generated_text[0].get("content", "").strip()
                        elif isinstance(generated_text, str):
                            # If string
                            proposal = generated_text.strip()
                        
                        # Format 2: {"content": "..."}
                        if not proposal:
                            proposal = element.get("content", "").strip()

                        logging.debug(f"Element {idx} (dict): extracted {len(proposal)} chars")

                    elif isinstance(element, str):
                        # Format 3: Direct string
                        proposal = element.strip()
                        logging.debug(f"Element {idx} (str): {len(proposal)} chars")

                    else:
                        logging.warning(f"Element {idx}: unknown type {type(element)}")
                        continue

                    if proposal:
                        proposals.append(proposal)
                        logging.info(f"Added proposal {len(proposals)}: {proposal[:80]}...")
            else:
                logging.error(f"Invalid response format: {type(response)}")
                return []

            logging.info(f"Total proposals extracted: {len(proposals)}")

            
            for proposal in proposals:
                logging.debug(f"Processing proposal: {proposal[:100]}...")

                
                if "Sub-question:" not in proposal:
                    logging.warning("Missing 'Sub-question:' marker, skipping")
                    continue

                
                try:
                    sub_query = proposal.split("Sub-question:", 1)[1].strip()
                except IndexError:
                    logging.error("Failed to split 'Sub-question:' marker")
                    continue

                
                if len(sub_query) < 10:
                    logging.warning(f"Sub-query too short ({len(sub_query)} chars): {sub_query}")
                    continue

                if sub_query in state:
                    logging.warning("Sub-query is duplicate of original query")
                    continue

                proposed_sub_queries.append(sub_query)
                logging.info(f"Valid sub-query extracted: {sub_query[:80]}...")

        except Exception as e:
            logging.error(f"Error processing sub-queries: {str(e)}", exc_info=True)
            return []

        
        if not proposed_sub_queries:
            logging.warning("No valid sub-queries extracted from responses")
            return []

        # ========== Step 4: Generate answers for each sub-query ==========
        revised_sub_queries = []

        for idx, sub_query in enumerate(proposed_sub_queries, 1):
            logging.info(f"Processing sub-query {idx}/{len(proposed_sub_queries)}: {sub_query[:60]}...")

            try:
                # Generate answer prompt
                sub_query_prompt, external_knowledge = answer_wrap(
                    sub_query, self.data_path, self.data_idx
                )
            except Exception as e:
                logging.error(f"Error generating answer prompt for sub-query: {str(e)}")
                continue

            # Get answer (with retry) - use sub-question and answer function (may be online model)
            response = []
            for attempt in range(1, max_retries + 1):
                try:
                    response = get_subq_and_answer_response(
                        sub_query_prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                        max_length=self.max_length,
                        truncation=self.truncation,
                        do_sample=self.do_sample,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=1,
                    )

                    # ===== Key fix: Uniformly check generated_text and content =====
                    if response and isinstance(response, list) and len(response) > 0:
                        content = ""

                        if isinstance(response[0], dict):
                            
                            generated_text = response[0].get("generated_text", "")
                            if isinstance(generated_text, list) and len(generated_text) > 0:
                                # If list, take first element's content
                                content = generated_text[0].get("content", "").strip()
                            elif isinstance(generated_text, str):
                                # If string
                                content = generated_text.strip()
                            else:
                                # Fallback check content field
                                content = response[0].get("content", "").strip()
                        elif isinstance(response[0], str):
                            content = response[0].strip()

                        if content:
                            break  # Got valid content
                        else:
                            logging.warning(f"Answer attempt {attempt}: Empty content")
                            response = []

                    if attempt < max_retries:
                        time.sleep(1 * attempt)

                except Exception as e:
                    logging.error(f"Answer attempt {attempt} failed: {str(e)}")
                    if attempt < max_retries:
                        time.sleep(1 * attempt)

            # Check if answer was obtained
            if not response:
                logging.warning(f"Failed to get answer for sub-query after {max_retries} retries")
                continue

            # ===== Key fix: Uniformly parse answer =====
            try:
                proposal = ""

                if isinstance(response[0], dict):
                   
                    generated_text = response[0].get("generated_text", "")
                    if isinstance(generated_text, list) and len(generated_text) > 0:
                        # If list, take first element's content
                        proposal = generated_text[0].get("content", "").strip()
                    elif isinstance(generated_text, str):
                        # If string
                        proposal = generated_text.strip()
                    else:
                        # Fallback check content field
                        proposal = response[0].get("content", "").strip()
                elif isinstance(response[0], str):
                    proposal = response[0].strip()
                else:
                    logging.error(f"Invalid answer format: {type(response[0])}")
                    continue

                
                if len(proposal) < 6:
                    logging.warning(f"Answer too short ({len(proposal)} chars)")
                    continue

                if "No directly relevant facts found" in proposal:
                    logging.info("No relevant facts found, skipping")
                    continue

                if proposal in state:
                    logging.warning("Answer duplicates existing state")
                    continue

                
                tmp_dict = {
                    sub_query: proposal,
                    "external_knowledge": external_knowledge
                }
                revised_sub_queries.append(tmp_dict)
                logging.info(f"Successfully added sub-query with answer (length: {len(proposal)})")

            except Exception as e:
                logging.error(f"Error processing answer: {str(e)}", exc_info=True)
                continue

        # ========== Return results ==========
        if not revised_sub_queries:
            logging.warning("No valid sub-query-answer pairs generated")
            return []

        logging.info(f"Successfully generated {len(revised_sub_queries)} sub-query nodes")
        return revised_sub_queries

    def get_node_value(self, query, answer):
        """
        Calculate weighted combination of:
        1-200. TF-IDF similarity between query-answer pair and original question
        2. TF-IDF similarity between knowledge and query to assess knowledge reliability

        Returns:
            float: Combined similarity score
        """
        if self.value_mode == "sim":
            return similarity_value(
                self.data["question"], query, answer, self.ans_weight
            )

        elif self.value_mode == "risk":
            return risk_value(self.data["question"], query, answer, self.ans_weight)

        elif self.value_mode == "random":
            return random.uniform(self.low_value, self.high_value)

        elif self.value_mode == "star":
            prompt = star_value_prompt.format(
                question=self.data["question"], sub=query, ans=answer
            )
            # Use value evaluation function (always uses local model)
            response = get_answer(prompt, self, 1)
            response = response[0] if response else ""
            try:
                score = float(response)
            except ValueError:
                score = 0.0
            return score

        else:
            raise ValueError(f"Invalid value mode: {self.value_mode}")

    def run(self):
        """
        Run MCTS search.

        Returns:
            TreeNode: Root node of the search tree
        """
        try:
            root_node, search_metric = MCTS_search(self)
            self.root_node = root_node  # Store for class-level access if needed
            print(f"Search completed with {search_metric} iterations")
            return root_node
        except Exception as e:
            logging.error(f"Error during MCTS search: {str(e)}")
            raise

    def traverse_tree(self, node):
        """
        Traverse the tree and save results to a JSON file.

        Args:
            node: Current node to traverse. If None, starts from root_node
            output_file: Path to output file
        """
        if node is None:
            node = self.root_node

        if node is None:
            return

        try:
            if node.children:
                for child in node.children:
                    self.traverse_tree(node.children[child])

            path_q = []
            cur_node = node
            while cur_node.parent is not None:
                path_q.append(cur_node.query)
                cur_node = cur_node.parent

            path_q.append(cur_node.query)

            final_answer_prompt, knowledge = final_answer_wrap(
                node.state, self.data_path, self.data_idx
            )

            max_retries = 3
            attempts = 0
            response = []

            while not response and attempts < max_retries:
                try:
                    # Use sub-question and answer function (may be online model)
                    response = get_subq_and_answer_response(
                        final_answer_prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                        max_length=self.max_length,
                        truncation=self.truncation,
                        do_sample=self.do_sample,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=1,
                    )
                except Exception as e:
                    logging.error(
                        f"Error getting model response (attempt {attempts + 1}): {str(e)}"
                    )
                attempts += 1

            if not response:
                logging.error("Failed to get reflection after all retries")
                return

            # Process response
            try:
                if (
                    isinstance(response, list)
                    and response
                    and isinstance(response[0], dict)
                ):
                    generated_text = response[0].get("generated_text", "")
                    if isinstance(generated_text, list) and len(generated_text) > 0:
                        # If list, take first element's content
                        proposal = generated_text[0].get("content", "")
                    elif isinstance(generated_text, str):
                        # If string
                        proposal = generated_text
                    else:
                        # Fallback check content
                        proposal = response[0].get("content", "")
                elif (
                    isinstance(response, list)
                    and response
                    and isinstance(response[0], str)
                ):
                    proposal = response[0]
                else:
                    logging.error("Invalid output format of reflection, not a list")
                    return

                final_answer = proposal

            except Exception as e:
                logging.error(f"Error processing model response: {str(e)}")
                return

            self.leaf_nodes.append(
                {
                    "original query": self.data["question"],
                    "last_query": node.state,
                    "answer": final_answer,
                    "facts": knowledge,
                    "path": path_q,
                    "value": node.value,
                }
            )

        except Exception as e:
            logging.error(f"Error while traversing tree: {str(e)}")
            raise

        return self.leaf_nodes

    def get_best_path(self):
        """
        Get the leaf node with the highest value.

        Returns:
            TreeNode: Leaf node with the highest value
        """
        current_node = self.root_node
        path_q = []
        while not current_node.is_terminal:
            if current_node.children:
                candidate_values = []
                for child in current_node.children.values():
                    candidate_values.append(child.value)
                best_candidate_idx = np.argmax(candidate_values)
                current_node = current_node.children[
                    list(current_node.children.keys())[best_candidate_idx]
                ]
            else:
                current_node.value = 0
                current_node = current_node.parent

        path_node = current_node
        while path_node.parent is not None:
            path_q.append(path_node.query)
            path_node = path_node.parent

        path_q.append(path_node.query)

        final_answer_prompt, knowledge = final_answer_wrap(
            current_node.state, self.data_path, self.data_idx
        )

        max_retries = 3
        attempts = 0
        response = []

        while not response and attempts < max_retries:
            try:
                # Use sub-question and answer function (may be online model)
                response = get_subq_and_answer_response(
                    final_answer_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    do_sample=self.do_sample,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                )
            except Exception as e:
                logging.error(
                    f"Error getting model response (attempt {attempts + 1}): {str(e)}"
                )
            attempts += 1

        if not response:
            logging.error("Failed to get reflection after all retries")
            return

            # Process response
        try:
            if (
                isinstance(response, list)
                and response
                and isinstance(response[0], dict)
            ):
                generated_text = response[0].get("generated_text", "")
                if isinstance(generated_text, list) and len(generated_text) > 0:
                    # If list, take first element's content
                    proposal = generated_text[0].get("content", "")
                elif isinstance(generated_text, str):
                    # If string
                    proposal = generated_text
                else:
                    # Fallback check content
                    proposal = response[0].get("content", "")
            elif (
                isinstance(response, list) and response and isinstance(response[0], str)
            ):
                proposal = response[0]
            else:
                logging.error("Invalid output format of reflection, not a list")
                return

            final_answer = proposal

        except Exception as e:
            logging.error(f"Error processing model response: {str(e)}")
            return

        result = {
            "original query": self.data["question"],
            "last_query": current_node.state,
            "answer": final_answer,
            "path": path_q,
            "facts": knowledge,
        }
        return result