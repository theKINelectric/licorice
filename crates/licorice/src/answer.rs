//! Answering (SPEC §9.7): the retrieved context plus an answer instruction, rendered as a
//! [`ModelRequest`]. The output is free text.

use licorice_providers::chat::ModelRequest;

use crate::retrieve::Retrieved;

pub const ANSWER_SYSTEM: &str = "You answer questions about people from your memory of past conversations with them. \
You answer briefly and only from that memory.";

const ANSWER_INSTRUCTIONS: &str = r#"Answer the question from the memory above.

How to read the memory: each fact starts with the date it happened, or "said <date>" when only the day it was mentioned is known. [D3:5] marks the source turns. Excerpts are raw dialogue, headed by the session date. Session summaries give the gist of whole sessions.

- Answer in a few words or one short sentence, with no preamble.
- For "when" questions, give an absolute date ("7 May 2023") or period ("the week before 9 June 2023"), worked out from the dates in memory.
- If several items in memory answer the question (for example, all the activities someone did), list them all.
- If the question asks what someone would likely do, think or prefer, give the most likely answer from what memory says about them.
- If memory does not contain the answer, or the question's premise contradicts memory (for example it says one person did what another did), reply "Not in memory" and briefly what memory does say."#;

/// The answer request for `question`, asked over `retrieved`.
pub fn request(id: impl Into<String>, question: &str, retrieved: &Retrieved) -> ModelRequest {
    ModelRequest {
        id: id.into(),
        task: "answer".into(),
        system: ANSWER_SYSTEM.into(),
        user: format!("{}\nQuestion: {question}\n\n{ANSWER_INSTRUCTIONS}", retrieved.context),
        schema: None,
        max_tokens: 200,
    }
}

/// Recover the exact memory from a request rendered by [`request`]. Used by the
/// evaluator to compare judges on saved contexts without retrieving again.
pub fn context_from_request<'a>(request: &'a ModelRequest, question: &str) -> Option<&'a str> {
    if request.task != "answer" || request.system != ANSWER_SYSTEM {
        return None;
    }
    request.user.strip_suffix(&format!("\nQuestion: {question}\n\n{ANSWER_INSTRUCTIONS}"))
}
