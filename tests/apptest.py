import unittest
from unittest.mock import patch
import requests_mock
from app import get_prompt, chat_response

class TestAppFunctions(unittest.TestCase):

    def test_get_prompt(self):
        # Test the function with example parameters
        instruction = "Example Instruction"
        examples = "Q: What is testing?\nA: "
        new_system_prompt = "You are an assistant."
        expected_output = "<<SYS>>\nYou are an assistant.\n<</SYS>>\n\nExample Instruction\nQ: What is testing?\nA: "
        result = get_prompt(instruction, examples, new_system_prompt)
        self.assertEqual(result, expected_output)

    @patch('app.qa_chain')  # Mock the qa_chain object in your app module
    def test_chat_response(self, mock_qa_chain):

        # Setup the mock to return a specific value
        mock_qa_chain.return_value = {"result": "Test response"}

        # Call the function with test data
        response = chat_response("Test message", {})
        self.assertEqual(response, "Test response")

        # Test handling of an empty or None message
        mock_qa_chain.return_value = {"result": "No input provided"}
        response = chat_response("", {})
        self.assertEqual(response, "No input provided")

        # You can add more tests to simulate different responses or errors

if __name__ == '__main__':
    unittest.main()
