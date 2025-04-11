import logging
from initialization.utils.llm_util import request_response

logger = logging.getLogger(__name__)

class ContextExpander:
    def __init__(self):
        self.role_prompt = (
            "You are an expert in Simulation-based Testing for Autonomous Driving Systems and in describing traffic accidents in detail. "
            "Your task is to expand a brief accident report by inserting additional, logically inferred prior events that occurred before the reported incident. "
            "For example, in the origin report, V1 was traveling westbound negotiating a left turn, you need to inserting prior events like "
            " V1 was traveling westbound at the right lane, then decelerated and changed lanes to the left. After that, V1 was negotiating a left turn. "
            "Do not remove or modify any information from the original report; instead, prepend additional context that explains the actions and movements of each vehicle leading up to the incident. "
            "Provide only the final expanded narrative without showing any internal reasoning or chain-of-thought."
        )

        self.task_prompt = (
            "Expand the following accident description by adding a series of detailed prior actions and events that logically occurred before the described incident. "
            "The final output should be a single, coherent narrative that begins with these additional details and then includes the original accident report unaltered. "
            "For each vehicle, describe their prior actions using only the following verbs: {brake, decelerate, accelerate, change lane left, change lane right, left turn (intersection), right turn (intersection), U turn (intersection)}. "
            "Do not include any chain-of-thought or extra commentary—only output the final expanded narrative."
        )

    def expand(self, brief_report):
        dialogue_history = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": brief_report + "\n\n" + self.task_prompt}
        ]

        try:
            response = request_response(dialogue_history, task_id=0)
            expanded_report = response.choices[0].message.content.strip()
            logger.info("Expanded Report: \n%s", expanded_report)
            return expanded_report
        except Exception as e:
            logger.error("Error generating expanded report: %s", str(e))
            return "Error: Unable to expand the report at this time."
