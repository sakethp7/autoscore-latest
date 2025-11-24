import streamlit as st
import requests
from PIL import Image
import io


API_URL = "http://13.60.36.216/autoscore/"  

st.set_page_config(
    page_title="AutoScore Dashboard",
    page_icon="📝",
    layout="wide"
)

def process_evaluation(question_text, question_files, answer_files):
    """Sends the request to the FastAPI backend."""
    
    # Prepare the multipart/form-data payload
    files = []
    
    # Add Answer Images (Required)
    for i, file in enumerate(answer_files):
        # Reset file pointer to beginning
        file.seek(0)
        # Format: ('answer_images', (filename, file_bytes, content_type))
        files.append(('answer_images', (file.name, file.read(), file.type)))
        
    # Add Question Images (Optional)
    if question_files:
        for i, file in enumerate(question_files):
            file.seek(0)
            files.append(('question_images', (file.name, file.read(), file.type)))
            
    # Add Form Data
    data = {
        'question': question_text
    }
    
    try:
        with st.spinner('Transcribing & Evaluating... Please wait.'):
            response = requests.post(API_URL, data=data, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(f"Connection refused. Is the backend running at `{API_URL}`?")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# --- UI LAYOUT ---
st.title("📝 Student Answer Evaluation System")
st.markdown("Upload student answer sheets and question details for instant AI grading.")

# Split layout: Inputs on Left (Sidebar-ish), Results on Right
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("1. Input Details")
    
    with st.form("evaluation_form"):
        q_text = st.text_area("Question Text (Required)", height=150, placeholder="Enter the full question text here...")
        
        st.markdown("---")
        st.markdown("**Reference Images (Optional)**")
        q_imgs = st.file_uploader("Upload Question/Diagram Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key="q_imgs")
        
        st.markdown("---")
        st.markdown("**Student Work (Required)**")
        ans_imgs = st.file_uploader("Upload Student Answer Sheets", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key="ans_imgs")
        
        submitted = st.form_submit_button("🚀 Evaluate Answer", use_container_width=True)

with col2:
    if submitted:
        if not q_text:
            st.warning("⚠️ Please provide the Question Text.")
        elif not ans_imgs:
            st.warning("⚠️ Please upload at least one Student Answer image.")
        else:
            # Process Request
            api_response = process_evaluation(q_text, q_imgs, ans_imgs)
            
            if api_response and api_response.get('status') == 'success':
                result = api_response.get('result', {})
                extracted_text = api_response.get('extracted_text', '')
                
                # --- DISPLAY RESULTS ---
                st.subheader("2. Evaluation Results")
                
                # Scorecard
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("Score Obtained", f"{result.get('score')} / {result.get('max_marks')}")
                with score_col2:
                    error_type = result.get('error_type', 'None').replace('_', ' ').title()
                    st.metric("Error Type", error_type)
                with score_col3:
                    time_analysis = result.get('time_analysis', 'N/A').title()
                    st.metric("Time Analysis", time_analysis)
                
                # Detailed Feedback
                st.info(f"**Concepts Required:** {result.get('concepts_required')}")
                
                with st.expander("🔍 Mistakes & Gap Analysis", expanded=True):
                    st.markdown(f"**Specific Mistakes:**")
                    st.write(result.get('mistakes_made'))
                    st.divider()
                    st.markdown(f"**Gap Analysis:**")
                    st.write(result.get('gap_analysis'))

                with st.expander("👨‍🏫 Teacher's Comments"):
                    st.write(result.get('additional_comments'))

                # Transcription Section
                st.subheader("3. Digitized Answer Sheet")
                st.markdown("The AI transcribed the handwritten image into text/LaTeX below:")
                
                # Clean up the <page> tags for display if desired, or keep raw
                display_text = extracted_text.replace("<page1>", "").replace("</page1>", "").strip()
                
                container = st.container(border=True)
                with container:
                    # Streamlit Markdown supports LaTeX via $...$
                    st.markdown(display_text)
                    
            elif api_response:
                st.error("Failed to parse response.")
