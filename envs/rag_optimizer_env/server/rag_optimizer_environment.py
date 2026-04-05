# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rag Optimizer Environment Implementation.
The agent acts as a Data Engineer to un-block a broken RAG pipeline.
"""

from uuid import uuid4
from typing import Dict, Any, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Import scikit-learn for our Grader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from models import RagOptimizerAction, RagOptimizerObservation
except ImportError:
    from models import RagOptimizerAction, RagOptimizerObservation


class RagOptimizerEnvironment(Environment):
    """
    RAG Optimizer Engine.
    Maintains a simulated Knowledge Base and grades it using TF-IDF.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Initial messy knowledge base
        self.kb = {
            "doc_pricing_legacy": {
                "text": "Pricing for 2021: Enterprise tier is $1000/mo. Standard is $500/mo. All plans include 10 users.",
                "metadata": {"type": "pricing"}
            },
            "doc_pricing_current_v2": {
                "text": "Current Pricing 2024: Enterprise is $1500/mo. Standard is $750/mo. Refunds are not permitted on the enterprise tier.",
                "metadata": {}
            },
            "doc_shipping_policy": {
                "text": "All internal shipments to remote branch offices take 5-7 business days. Overnight shipping is only available for C-suite.",
                "metadata": {"department": "logistics"}
            },
            "doc_messy_support_ticket_1": {
                "text": "User complained the button disappeared on the frontend. Another user said the database latency was high. The frontend team fixed the button by updating CSS.",
                "metadata": {}
            },
            "doc_messy_support_ticket_2": {
                "text": "Email integration is failing with error 401 Unauthorized. The API key was rotated on Tuesday.",
                "metadata": {}
            },
            "doc_monolithic_onboarding": {
                "text": "Welcome to the company! Here are some rules. 1) VPN access requires DUO. 2) The cafetaria opens at 8 AM. 3) For HR issues, email hr@company.com. 4) The 2024 holiday schedule includes Dec 25, Jan 1, and July 4. 5) Parking passes must be renewed annually in March.",
                "metadata": {}
            },
            # Add distractor files
            **{f"doc_distractor_hr_{i}": {"text": f"This is an old HR policy document regarding {['pto', 'sick leave', 'travel', 'expenses'][i%4]} from 201{i%10}.", "metadata":{}} for i in range(10)},
            **{f"doc_distractor_eng_{i}": {"text": f"Engineering architecture decision record {i}. We decided to use {['React', 'Postgres', 'Redis', 'Kafka'][i%4]} because of scaling concerns.", "metadata":{}} for i in range(10)},
            **{f"doc_distractor_random_{i}": {"text": f"Weekly team update notes. Nothing important here, just discussed the weather and the upcoming launch {i}.", "metadata":{}} for i in range(10)},
        }
        
        # Hidden test suite for the grader
        self.test_suite = [
            {
                "query": "What is the current 2024 price for standard?",
                "target_concept": "750/mo"
            },
            {
                "query": "What is the refund policy for enterprise?",
                "target_concept": "Refunds are not permitted"
            },
            {
                "query": "UI issues frontend CSS missing button",
                "target_concept": "frontend team fixed the button"
            },
            {
                "query": "How long does shipping take to branch offices?",
                "target_concept": "5-7 business days"
            },
            {
                "query": "What months do parking passes need to be renewed?",
                "target_concept": "March"
            },
            {
                "query": "What holidays are we off in 2024?",
                "target_concept": "July 4"
            }
        ]

    def _get_kb_summary(self) -> Dict[str, Dict]:
        """Returns a summary of the KB for the observation."""
        summary = {}
        for k, v in self.kb.items():
            summary[k] = {"metadata": v.get("metadata", {}), "length": len(v.get("text", ""))}
        return summary

    def reset(self) -> RagOptimizerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return RagOptimizerObservation(
            message="RagOptimizerEnv Initialized. You have messy chunks in the KB. Resolve conflicts, add metadata tags to short tickets, and splinter monolithic files to win.",
            current_docs=self._get_kb_summary(),
            done=False,
            reward=0.0
        )

    def _evaluate_kb(self) -> float:
        """The Grader: Evaluates the agent's current KB using TF-IDF."""
        if not self.kb:
            return 0.0
            
        doc_texts = [doc["text"] for doc in self.kb.values()]
        
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            doc_vectors = vectorizer.fit_transform(doc_texts)
        except ValueError:
            return 0.0
            
        score = 0.0
        
        for case in self.test_suite:
            query_vec = vectorizer.transform([case["query"]])
            similarities = cosine_similarity(query_vec, doc_vectors)[0]
            
            # Get top 3
            top_k_indices = similarities.argsort()[-3:][::-1]
            
            found = False
            for idx in top_k_indices:
                if similarities[idx] > 0.01:
                    if case["target_concept"].lower() in doc_texts[idx].lower():
                        found = True
                        break
            if found:
                score += 1.0
                
        return float(score / len(self.test_suite))

    def step(self, action: RagOptimizerAction) -> RagOptimizerObservation:  # type: ignore[override]
        self._state.step_count += 1
        
        msg = ""
        done = False
        reward = 0.0
        
        try:
            if action.action_type == "read_document":
                if action.doc_id in self.kb:
                    msg = f"Content of {action.doc_id}: {self.kb[action.doc_id]['text']}"
                else:
                    msg = f"Error: doc_id {action.doc_id} not found."
            
            elif action.action_type == "delete_document":
                if action.doc_id in self.kb:
                    del self.kb[action.doc_id]
                    msg = f"Deleted {action.doc_id}."
                else:
                    msg = f"Error: doc_id {action.doc_id} not found."
                    
            elif action.action_type == "update_document":
                if not action.doc_id or not action.text:
                    msg = "Error: doc_id and text required for update_document."
                else:
                    if action.doc_id not in self.kb:
                        self.kb[action.doc_id] = {"text": "", "metadata": {}}
                    self.kb[action.doc_id]["text"] = action.text
                    msg = f"Updated text for {action.doc_id}."
                    
            elif action.action_type == "add_metadata":
                if not action.doc_id or not action.metadata_key or not action.metadata_value:
                    msg = "Error: doc_id, metadata_key, and metadata_value required."
                else:
                    if action.doc_id not in self.kb:
                        msg = f"Error: doc_id {action.doc_id} not found."
                    else:
                        self.kb[action.doc_id]["metadata"][action.metadata_key] = action.metadata_value
                        msg = f"Added metadata to {action.doc_id}."
                        
            elif action.action_type == "submit":
                done = True
                reward = self._evaluate_kb()
                msg = f"Evaluation complete. Final reward: {reward:.2f}"
                
        except Exception as e:
            msg = f"Action failed: {str(e)}"

        return RagOptimizerObservation(
            message=msg,
            current_docs=self._get_kb_summary(),
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
