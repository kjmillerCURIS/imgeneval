new_prompt = ('''
          You are an AI assistant that receives a set of yes/no question–answer pairs about an image, along with ground-truth or model-predicted answers. Your tasks are:
          Perform a consistency check on the question–answer pairs. If answers are logically contradictory, you must resolve or discard the conflicting True item(s).
          Example Rule: If “Are there [X] in the image?” is answered “False,” then any claim like “Are the [X] doing Y?” cannot be “True.”
          Example Rule: If “Is [ACTION] happening?” is “False,” then any sub-question (e.g., “Who is doing [ACTION]?”) must also be “False.”
          Output a final “reconciled” set of question–answer pairs where contradictory answers have been removed or corrected.
          Generate a short factual summary (one or two sentences) that reflects only the remaining True statements after reconciliation.
          Do not include details marked false or discarded.
          Keep it concise and do not restate the entire prompt or caption verbatim.
          Emphasize important relations (e.g., who is doing what to whom, if relevant).
          Example
          Input Q–A pairs:
          “Are there leaves in the image?” → False
          “Are leaves being shed?” → True
          “Is there a cat in the image?” → False
          “Are the leaves falling upwards?” → False
          Step 1 (Consistency): Realize #1 and #2 contradict each other. You cannot have leaves being shed if there are no leaves.
          Choose to keep #1 as likely correct (False = no leaves). Force #2 to False or discard it.
          Step 2 (Reconciled Answers):“Are there leaves in the image?” → False
          “Are leaves being shed?” → False (changed from True)
          “Is there a cat in the image?” → False
          “Are the leaves falling upwards?” → False
          Step 3 (Summary):Since all final answers are False, the summary might be “No leaves or cats are present,” or simply “No relevant objects or actions confirmed.”
          You will apply similar logic for any set of Q–A pairs, then produce the final reconciled set plus a concise summary.
          Give your answer in the following format: #Answer: [The summary]
            '''+ "\n".join(f"{item} (True)" for item in checklist_trues)
             + "\n".join(f"{item} (False)" for item in checklist_false)
            + "\n")
