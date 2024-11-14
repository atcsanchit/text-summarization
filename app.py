import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.pipeline.prediction_pipeline import PredictionPipeline


@st.cache_resource
# def load_model():
#     model_name = "microsoft/DialoGPT-medium"  
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     return model, tokenizer

# model, tokenizer = load_model()


def get_response(user_input):

    # new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')


    # bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1) if history is not None else new_user_input_ids


    # chat_history_ids = model.generate(
    #     bot_input_ids,
    #     max_length=1000,
    #     pad_token_id=tokenizer.eos_token_id,
    #     num_return_sequences=1
    # )


    # response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    # return response, chat_history_ids

    prediction_obj = PredictionPipeline()
    response = prediction_obj.initiate_pipeline(text=user_input)
    return response



st.title("Chatbot Application")
st.markdown("This is a simple chatbot using a pre-trained model.")


# if 'history' not in st.session_state:
#     st.session_state.history = None

user_input = st.text_input("You:", "")

if user_input:
    with st.spinner("Generating response..."):
        response = get_response(user_input)
        st.write(f"Bot: {response}")


# if st.session_state.history is not None and len(st.session_state.history) > 0:
#     st.markdown("### Conversation History")
    # history_text = tokenizer.decode(st.session_state.history[0], skip_special_tokens=True)
    # st.text_area("Chat History", history_text, height=200)
