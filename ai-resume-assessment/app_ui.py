import os
import requests
import streamlit as st

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="AI-Driven Resume Assessment System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .skill-chip {
        display: inline-block;
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 0.3rem 0.65rem;
        border-radius: 12px;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 0.2rem;
        border: 1px solid #BBDEFB;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .score-badge {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://127.0.0.1:8000"

# Endpoints
PARSE_ENDPOINT = f"{API_BASE_URL}/api/v1/resume/parse"
SAVE_CANDIDATE_ENDPOINT = f"{API_BASE_URL}/api/v1/candidate/save"
START_ASSESSMENT_ENDPOINT = f"{API_BASE_URL}/api/v1/assessment/start"
CHAT_ASSESSMENT_ENDPOINT = f"{API_BASE_URL}/api/v1/assessment/chat"
RESULTS_ENDPOINT = f"{API_BASE_URL}/api/v1/assessment/results"
RANK_ENDPOINT = f"{API_BASE_URL}/api/v1/candidate/rank"


# 2. Backend Health Check
def check_backend_status():
    try:
        res = requests.get(f"{API_BASE_URL}/docs", timeout=3)
        if res.status_code == 200:
            return True, "Connected (FastAPI Active)"
        return False, f"Degraded (HTTP {res.status_code})"
    except Exception:
        return False, "Offline (Backend Server Not Found)"


st.sidebar.title("⚙️ System Status")
is_connected, status_text = check_backend_status()

if is_connected:
    st.sidebar.success(f"Backend API: {status_text}")
else:
    st.sidebar.error(f"Backend API: {status_text}")
    st.sidebar.info("💡 Start backend server:\n`python -m uvicorn app.main:app --reload`")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 System Architecture Modules")
st.sidebar.markdown("""
1. **Phase 1**: Resume PDF Parsing & File Storage
2. **Phase 2**: Candidate DB & Qdrant Vector Index
3. **Phase 3**: Dynamic Multi-Phase AI Interview Engine
4. **Phase 4**: LLM Judge Scorecard & Semantic Ranking
""")

# Main Header
st.markdown('<div class="main-title">📄 AI-Driven Resume Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End AI recruitment pipeline: PDF Parsing, Qdrant Vector Search, Adaptive AI Technical Interviewing, and LLM Scorecards.</div>', unsafe_allow_html=True)

# Main Navigation Tabs for the 4 Modules
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Module 1: Resume Parsing",
    "💾 Module 2: DB & Vector Storage",
    "💬 Module 3: AI Interview Engine",
    "🏆 Module 4: Scorecard & Ranking"
])


# ==============================================================================
# MODULE 1: Resume Parsing & Dynamic File Storage
# ==============================================================================
with tab1:
    st.header("📄 Module 1: PDF Resume Parsing & Production Persistence")
    st.caption("Upload a candidate PDF resume to parse structured data and persist with dynamic production filenames.")

    col_upload, col_action = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"], key="mod1_pdf")

    with col_action:
        st.write(" ")
        st.write(" ")
        parse_btn = st.button("🚀 Parse & Extract Resume", use_container_width=True, type="primary", disabled=not uploaded_file)

    if parse_btn and uploaded_file:
        with st.spinner("Extracting text and parsing profile with Gemini AI..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(PARSE_ENDPOINT, files=files, timeout=45)

                if response.status_code == 200:
                    result = response.json()
                    st.session_state["parsed_data"] = result
                    st.success("Resume parsed and saved successfully!")
                else:
                    err_msg = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                    st.error(f"Parsing failed (HTTP {response.status_code}): {err_msg}")
            except Exception as e:
                st.error(f"Connection error to FastAPI backend: {str(e)}")

    if "parsed_data" in st.session_state:
        data = st.session_state["parsed_data"]
        profile = data.get("parsed_profile", {})
        gaps = data.get("missing_gaps", [])

        st.markdown("---")
        st.subheader("👤 Candidate Profile Overview")

        saved_filename = data.get("saved_filename")
        if saved_filename:
            st.info(f"📁 **Saved Production File**: `{saved_filename}` | **Path**: `{data.get('storage_path', 'uploads/resumes/')}`")

        personal_info = profile.get("personal_info", {})
        c_name = profile.get("candidate_name") or personal_info.get("name") or "Not Specified"
        c_email = profile.get("email") or personal_info.get("email") or "Not Specified"
        c_phone = profile.get("phone") or personal_info.get("phone") or "Not Specified"

        m1, m2, m3 = st.columns(3)
        m1.metric("Candidate Name", c_name)
        m2.metric("Email Address", c_email)
        m3.metric("Phone Number", c_phone)

        st.markdown("### 💡 Core Technical Skills")
        core_skills = profile.get("core_skills", [])
        if core_skills:
            badges = "".join([f'<span class="skill-chip">{s}</span>' for s in core_skills])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("No core technical skills identified.")

        st.markdown(" ")
        t_exp, t_edu, t_proj = st.tabs(["💼 Work Experience", "🎓 Education", "🚀 Projects"])

        with t_exp:
            work_exp = profile.get("work_experience", [])
            if work_exp:
                for idx, exp in enumerate(work_exp, 1):
                    role = exp.get("role") or "Role N/A"
                    company = exp.get("company") or "Company N/A"
                    dur = exp.get("duration") or ""
                    with st.expander(f"**{role}** at **{company}** ({dur})", expanded=(idx == 1)):
                        st.write(exp.get("description") or "No detailed description.")
            else:
                st.info("No work experience recorded.")

        with t_edu:
            edu_list = profile.get("education", [])
            if edu_list:
                for edu in edu_list:
                    deg = edu.get("degree") or "Degree N/A"
                    inst = edu.get("institution") or "Institution N/A"
                    yr = edu.get("year") or "Year N/A"
                    st.write(f"- **{deg}** — {inst} (Year: {yr})")
            else:
                st.info("No education details recorded.")

        with t_proj:
            projects = profile.get("projects", [])
            if projects:
                for proj in projects:
                    p_name = proj.get("name") or "Project"
                    p_stack = ", ".join(proj.get("tech_stack", []))
                    st.write(f"**{p_name}** ({p_stack}): {proj.get('summary', '')}")
            else:
                st.info("No projects recorded.")

        st.markdown("---")
        st.subheader("⚠️ Profile Gaps & Recommendations")
        if gaps:
            for gap in gaps:
                if isinstance(gap, dict):
                    st.warning(f"• **[{gap.get('category', 'Gap')}]** {gap.get('issue', str(gap))} *(Severity: {gap.get('severity', 'Medium')})*")
                else:
                    st.warning(f"• {gap}")
        else:
            st.success("🎉 Profile is complete with no missing gaps!")


# ==============================================================================
# MODULE 2: Candidate DB & Qdrant Vector Storage
# ==============================================================================
with tab2:
    st.header("💾 Module 2: Candidate SQL Persistence & Qdrant Vector Indexing")
    st.caption("Save parsed candidate profiles to SQL database and index 1536-dimensional embeddings in Qdrant Vector DB.")

    if "parsed_data" not in st.session_state:
        st.warning("⚠️ Please upload and parse a resume in Module 1 first before saving candidate profile.")
    else:
        parsed_data = st.session_state["parsed_data"]
        prof = parsed_data.get("parsed_profile", {})
        p_info = prof.get("personal_info", {})
        c_name = prof.get("candidate_name") or p_info.get("name") or "Candidate"
        c_email = prof.get("email") or p_info.get("email") or "candidate@example.com"

        st.subheader("📋 Candidate Profile Payload to Persist")
        st.json({
            "name": c_name,
            "email": c_email,
            "skills_count": len(prof.get("core_skills", [])),
            "detected_gaps": len(parsed_data.get("missing_gaps", []))
        })

        if st.button("💾 Save Profile to SQL DB & Index Vector in Qdrant", type="primary", use_container_width=True):
            with st.spinner("Saving profile to SQL DB & generating Qdrant vector embedding..."):
                try:
                    payload = {
                        "name": c_name,
                        "email": c_email,
                        "parsed_profile": prof,
                        "missing_gaps": parsed_data.get("missing_gaps", [])
                    }
                    res = requests.post(SAVE_CANDIDATE_ENDPOINT, json=payload, timeout=30)
                    if res.status_code in [200, 201]:
                        save_res = res.json()
                        st.session_state["saved_candidate"] = save_res
                        st.session_state["candidate_id"] = save_res.get("candidate_id")
                        st.success("🎉 Candidate successfully saved to Database & Indexed in Qdrant!")
                    else:
                        st.error(f"Failed to save candidate (HTTP {res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")

        if "saved_candidate" in st.session_state:
            sc = st.session_state["saved_candidate"]
            st.markdown("---")
            st.subheader("✅ Database Persistence Confirmation")
            col1, col2, col3 = st.columns(3)
            col1.metric("Candidate ID", sc.get("candidate_id", "N/A"))
            col2.metric("Profile DB ID", sc.get("profile_id", "N/A"))
            col3.metric("Qdrant Vector Indexed", "True ✅" if sc.get("vector_indexed") else "False ❌")


# ==============================================================================
# MODULE 3: Dynamic Multi-Phase AI Technical Interview Engine
# ==============================================================================
with tab3:
    st.header("💬 Module 3: Dynamic Multi-Phase AI Technical Interview")
    st.caption("Conducts adaptive multi-phase technical interviews: Phase A (Gap Filling), Phase B (Skill Validation), Phase C (Problem Solving).")

    if "candidate_id" not in st.session_state:
        st.warning("⚠️ Please save candidate profile in Module 2 first to obtain a Candidate ID.")
    else:
        candidate_id = st.session_state["candidate_id"]
        st.info(f"Active Candidate ID: `{candidate_id}`")

        # Start Interview Button
        if "assessment_id" not in st.session_state:
            if st.button("🚀 Start Dynamic AI Interview Session", type="primary", use_container_width=True):
                with st.spinner("Initializing AI interview session & generating Phase 1 question..."):
                    try:
                        res = requests.post(START_ASSESSMENT_ENDPOINT, json={"candidate_id": candidate_id}, timeout=30)
                        if res.status_code in [200, 201]:
                            start_data = res.json()
                            st.session_state["assessment_id"] = start_data["assessment_id"]
                            st.session_state["current_phase"] = start_data["phase"]
                            st.session_state["current_question_id"] = start_data["question_id"]
                            st.session_state["current_question"] = start_data["question"]
                            st.session_state["chat_history"] = []
                            st.rerun()
                        else:
                            st.error(f"Failed to start assessment: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

        else:
            assessment_id = st.session_state["assessment_id"]
            
            # Safe phase resolution with fallback defaults so it never displays 'None'
            phase = st.session_state.get("current_phase")
            if not phase or phase in ["None", "null", ""]:
                phase = "Phase B (Skill Validation)"
            
            q_id = st.session_state.get("current_question_id")
            question = st.session_state.get("current_question", "")

            st.markdown(f"### 📍 Current Phase: **{phase}** (Assessment ID: `{assessment_id}`)")

            # Display previous Q&A turns
            if st.session_state.get("chat_history"):
                for idx, turn in enumerate(st.session_state["chat_history"], 1):
                    with st.chat_message("assistant"):
                        st.markdown(f"**Question {idx} [{turn.get('phase', 'Phase')}]:** {turn['question']}")
                    with st.chat_message("user"):
                        st.markdown(turn["answer"])
                    if "last_evaluation" in turn and turn["last_evaluation"]:
                        ev = turn["last_evaluation"]
                        t_acc = ev.get('technical_accuracy', 'N/A')
                        d_clr = ev.get('depth_clarity', 'N/A')
                        p_solv = ev.get('problem_solving_logic', 'N/A')
                        t_score = ev.get('turn_score', 'N/A')
                        expl = ev.get('explanation', '')
                        st.info(f"📊 **LLM-as-a-Judge Grade**: Accuracy: {t_acc}/10 | Clarity: {d_clr}/10 | Logic: {p_solv}/10 | **Turn Score**: **{t_score}/10**\n\n💡 *Feedback*: {expl}")

            # Active Question Box
            if question and not st.session_state.get("interview_completed"):
                with st.chat_message("assistant"):
                    st.markdown(f"**AI Interviewer Question:**\n{question}")

                # Answer Input Box with dynamic key for clean clearing on submit
                form_key = f"answer_form_{q_id or 'default'}"
                with st.form(key=form_key):
                    user_answer = st.text_area("Your Technical Answer:", placeholder="Enter your technical response here...", height=120)
                    submit_answer = st.form_submit_button("📤 Submit Technical Answer", type="primary")

                if submit_answer and user_answer.strip():
                    with st.spinner("LLM-as-a-Judge is evaluating your response and advancing interview..."):
                        try:
                            payload = {
                                "candidate_id": candidate_id,
                                "assessment_id": assessment_id,
                                "question_id": q_id,
                                "candidate_answer": user_answer.strip()
                            }
                            res = requests.post(CHAT_ASSESSMENT_ENDPOINT, json=payload, timeout=45)
                            if res.status_code == 200:
                                chat_res = res.json()
                                
                                # Retrieve next question, next phase, evaluation scores, and status
                                next_phase_val = chat_res.get("next_phase") or chat_res.get("phase") or "Phase B (Skill Validation)"
                                next_question_val = chat_res.get("next_question")
                                next_q_id_val = chat_res.get("next_question_id")
                                last_eval_val = chat_res.get("last_evaluation") or {}
                                status_val = chat_res.get("status", "in_progress")

                                # 1. Append turn history
                                st.session_state["chat_history"].append({
                                    "phase": phase,
                                    "question": question,
                                    "answer": user_answer.strip(),
                                    "last_evaluation": last_eval_val
                                })

                                # 2. Update session state with next phase & next question
                                if status_val == "completed" or not next_question_val:
                                    st.session_state["current_phase"] = "Completed"
                                    st.session_state["current_question"] = None
                                    st.session_state["current_question_id"] = None
                                    st.session_state["interview_completed"] = True
                                else:
                                    st.session_state["current_phase"] = next_phase_val
                                    st.session_state["current_question"] = next_question_val
                                    st.session_state["current_question_id"] = next_q_id_val
                                    st.session_state["interview_completed"] = False

                                # 3. Trigger immediate rerun to update UI state
                                st.rerun()
                            else:
                                st.error(f"Evaluation error (HTTP {res.status_code}): {res.text}")
                        except Exception as e:
                            st.error(f"Error submitting answer: {str(e)}")

            elif st.session_state.get("interview_completed"):
                st.success("🏆 All interview phases completed! Proceed to Module 4 to view complete scorecard results.")



# ==============================================================================
# MODULE 4: LLM Scorecard & Semantic Candidate Ranking
# ==============================================================================
with tab4:
    st.header("🏆 Module 4: Candidate Scorecard & Semantic Qdrant Ranking")
    st.caption("View LLM-as-a-Judge evaluations and execute semantic searches to rank candidates against Job Descriptions.")

    m4_sub1, m4_sub2 = st.tabs(["📊 Candidate Performance Scorecard", "🔍 Semantic Job Ranking"])

    # Sub-tab 1: Scorecard
    with m4_sub1:
        c_id = st.session_state.get("candidate_id")
        if not c_id:
            c_id = st.text_input("Enter Candidate ID to view scorecard:", value="")

        if c_id:
            if st.button("📊 Fetch Candidate Scorecard", type="primary"):
                with st.spinner("Fetching scorecard from server..."):
                    try:
                        res = requests.get(f"{RESULTS_ENDPOINT}/{c_id}", timeout=20)
                        if res.status_code == 200:
                            scorecard = res.json()
                            st.session_state["scorecard"] = scorecard
                        else:
                            st.error(f"Failed to fetch scorecard: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

        if "scorecard" in st.session_state:
            sc = st.session_state["scorecard"]
            st.markdown("---")
            st.subheader(f"📄 Assessment Scorecard: **{sc.get('name', 'Candidate')}** ({sc.get('email', '')})")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Overall Score (out of 10)", f"{sc.get('overall_score_out_of_10', 0)} / 10")
            with c2:
                st.metric("Percentile Rank", f"{sc.get('percentile_rank', 0)}%")

            st.markdown("### 📝 Written Feedback Justification")
            st.text_area("Detailed Feedback", value=sc.get("written_feedback", "No feedback recorded."), height=150, disabled=True)

            st.markdown("### 🔍 Evaluation Turns Breakdown")
            turns = sc.get("turns_breakdown", [])
            if turns:
                for idx, t in enumerate(turns, 1):
                    with st.expander(f"Turn #{idx} [{t.get('phase', 'Phase')}] — Turn Score: {t.get('turn_score')}/10"):
                        st.write(f"**Question**: {t.get('question')}")
                        st.write(f"**Candidate Answer**: {t.get('answer')}")
                        st.write(f"- Technical Accuracy: {t.get('technical_accuracy')}/10")
                        st.write(f"- Depth & Clarity: {t.get('depth_clarity')}/10")
                        st.write(f"- Problem Solving: {t.get('problem_solving_logic')}/10")
                        st.write(f"**Feedback Explanation**: {t.get('explanation')}")
            else:
                st.info("No turns recorded for this candidate.")

    # Sub-tab 2: Semantic Candidate Ranking
    with m4_sub2:
        st.subheader("🔍 Semantic Candidate Matcher & Ranker")
        st.caption("Ranks stored candidates by combining Qdrant Cosine vector similarity (60%) with interview performance score (40%).")

        jd_text = st.text_area(
            "Target Job Description:",
            value="Principal Software Architect with expertise in Python, FastAPI, async microservices, PostgreSQL, and Qdrant vector retrieval.",
            height=100
        )

        top_k = st.slider("Top K Candidates to Return:", min_value=1, max_value=20, value=5)

        if st.button("🚀 Rank Candidates for Job Description", type="primary", use_container_width=True):
            with st.spinner("Encoding Job Description vector & searching Qdrant collection..."):
                try:
                    res = requests.get(f"{RANK_ENDPOINT}?job_description={requests.utils.quote(jd_text)}&top_k={top_k}", timeout=30)
                    if res.status_code == 200:
                        rank_data = res.json()
                        st.session_state["rank_data"] = rank_data
                        st.success(f"Ranked {rank_data.get('total_ranked', 0)} candidate(s) successfully!")
                    else:
                        st.error(f"Ranking failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

        if "rank_data" in st.session_state:
            rd = st.session_state["rank_data"]
            candidates = rd.get("ranked_candidates", [])
            st.markdown("---")
            st.subheader(f"📊 Ranked Candidates ({len(candidates)} Result(s))")

            if candidates:
                for r_idx, c in enumerate(candidates, 1):
                    with st.expander(f"🏆 **Rank #{r_idx}: {c.get('name')}** — Composite Score: **{c.get('composite_score')}**", expanded=(r_idx == 1)):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Composite Match Score", f"{c.get('composite_score')}")
                        col_b.metric("Qdrant Semantic Match", f"{c.get('semantic_score')}")
                        col_c.metric("Interview Score", f"{c.get('assessment_score', 0)} / 10")

                        skills = c.get("skills", [])
                        if skills:
                            badges = "".join([f'<span class="skill-chip">{s}</span>' for s in skills])
                            st.markdown(f"**Extracted Skills**: {badges}", unsafe_allow_html=True)
            else:
                st.info("No candidate matches found.")
