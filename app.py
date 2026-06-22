"""
app.py
Gradio UI for the Jensen Huang Digital Twin (basic RAG version).

Layout:
  - Main chat window (left)
  - Sidebar (right): user id, era filter, last retrieved sources,
    long-term memory viewer, save/reset session controls

Written for Gradio 6.x: gr.Chatbot uses the "messages" format by default
(list of {"role": ..., "content": ...} dicts), and theme/css are passed
to demo.launch() rather than the Blocks constructor.
"""

import gradio as gr

from agent import JensenAgent
from memory import get_all_memories, delete_all_memories
import tts as tts_module

VOICE_AVAILABLE = tts_module.is_available()

ERA_OPTIONS = {
    "All Eras": "all",
    "Pre-CUDA (<= 2007)": "pre_cuda",
    "Deep Learning (2012-2021)": "deep_learning",
    "LLM Era (2022+)": "llm_era",
}


def get_agent(agent: JensenAgent | None, user_id: str) -> JensenAgent:
    user_id = (user_id or "default_user").strip() or "default_user"
    if agent is None or agent.user_id != user_id:
        agent = JensenAgent(user_id=user_id)
    return agent


def respond(message, history, agent, era_label, user_id, voice_enabled):
    if not message or not message.strip():
        return history, agent, gr.update(), gr.update(), gr.update(), gr.update()

    agent = get_agent(agent, user_id)
    agent.set_era(ERA_OPTIONS.get(era_label, "all"))

    result = agent.chat(message)

    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["answer"]})

    if result["sources"]:
        sources_md = "**Sources used this turn:**\n" + "\n".join(
            f"- {s}" for s in result["sources"]
        )
    else:
        sources_md = "_No sources retrieved this turn._"

    badge = f"`{result['query_type']}` · {result['chunks_used']} chunk(s) retrieved"

    audio_path = None
    tts_status = ""
    if voice_enabled:
        audio_path = tts_module.speak(result["answer"])
        if audio_path is None:
            tts_status = f"⚠️ Voice error: {tts_module.get_error()}"

    return history, agent, sources_md, badge, audio_path, tts_status


def save_session(agent, user_id):
    agent = get_agent(agent, user_id)
    agent.end_session()
    return agent, "Session saved — facts extracted into long-term memory."


def reset_chat(agent, user_id):
    agent = get_agent(agent, user_id)
    agent.reset_conversation()
    return agent, []


def load_memories(user_id):
    facts = get_all_memories((user_id or "default_user").strip() or "default_user")
    if not facts:
        return "_No long-term memories yet for this user._"
    lines = [f"- **[{f['category']}]** {f['fact']}" for f in facts]
    return "\n".join(lines)


def clear_memories(user_id):
    delete_all_memories((user_id or "default_user").strip() or "default_user")
    return "_Memories cleared._"


with gr.Blocks(title="Jensen Huang - Digital Twin") as demo:
    agent_state = gr.State(None)

    gr.Markdown("# Jensen Huang — Digital Twin", elem_id="jensen-title")
    gr.Image(
        value="jensen.webp",
        show_label=False,
        width=250,
        interactive=False
    )
    gr.Markdown(
        "Co-founder & CEO, NVIDIA · Stanford MS EE · Father of the GPU computing era"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, label="Chat with Jensen",  avatar_images=(None, "jensen.webp"))
            badge_md = gr.Markdown("")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask Jensen anything...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                reset_btn = gr.Button("Reset Chat")
                save_btn = gr.Button("Save Session")
            save_status = gr.Markdown("")
            audio_out = gr.Audio(
                label="Jensen's voice",
                autoplay=True,
                visible=VOICE_AVAILABLE,
            )
            tts_status_md = gr.Markdown("")

        with gr.Column(scale=1, elem_id="sidebar-box"):
            gr.Markdown("### ⚙️ Settings")
            user_id_box = gr.Textbox(label="User ID", value="default_user")
            era_radio = gr.Radio(
                list(ERA_OPTIONS.keys()), value="All Eras", label="Jensen speaks from"
            )

            if VOICE_AVAILABLE:
                voice_toggle = gr.Checkbox(
                    label="Enable Jensen's Voice (Qwen3-TTS clone)",
                    value=False,
                )
            else:
                gr.Markdown(
                    "_🔊 Voice cloning unavailable — `jensen_ref.wav` not found. "
                    "Run `data_collector.py` to extract it._"
                )
                voice_toggle = gr.Checkbox(value=False, visible=False)

            gr.Markdown("###  Last Retrieved Sources")
            sources_md = gr.Markdown("_No sources yet._")

            gr.Markdown("### Long-Term Memory")
            with gr.Row():
                load_mem_btn = gr.Button("Load Memories")
                clear_mem_btn = gr.Button("Clear All")
            memory_md = gr.Markdown("")

    chat_inputs = [msg, chatbot, agent_state, era_radio, user_id_box, voice_toggle]
    chat_outputs = [chatbot, agent_state, sources_md, badge_md, audio_out, tts_status_md]

    msg.submit(respond, chat_inputs, chat_outputs).then(lambda: "", None, msg)
    send_btn.click(respond, chat_inputs, chat_outputs).then(lambda: "", None, msg)

    reset_btn.click(reset_chat, [agent_state, user_id_box], [agent_state, chatbot])
    save_btn.click(save_session, [agent_state, user_id_box], [agent_state, save_status])

    load_mem_btn.click(load_memories, [user_id_box], [memory_md])
    clear_mem_btn.click(clear_memories, [user_id_box], [memory_md])


if __name__ == "__main__":
    demo.launch(share=True)
