import streamlit as st

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Course Recommendation AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR STYLING AND INTERACTIVITY ---
st.markdown("""
<style>
    /* General Dark Theme Adjustments for Clarity */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Style for the skill boxes (same as previous response) */
    .skill-box {
        background-color: #1f2a24; 
        color: #8dcf92; 
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 8px;
        border: 1px solid #28502f;
        font-size: 14px;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        height: 100%; 
    }
    .skill-box::before {
        content: '✓'; 
        color: #4CAF50;
        font-size: 18px;
    }

    /* Streamlit's default button styling is complex. 
       To achieve the desired hover/active effect easily with a cleaner layout, 
       we'll inject custom HTML links disguised as buttons for the sidebar. */

    .sidebar-link {
        display: flex;
        align-items: center;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 8px;
        text-decoration: none; /* Remove underline from link */
        color: white;
        font-weight: 500;
        transition: background-color 0.2s, color 0.2s; /* Smooth transition for animation effect */
        border: 1px solid transparent;
    }

    /* Hover effect: Fills with a light blue/accent color */
    .sidebar-link:hover {
        background-color: #1a4f83; /* Light blue/dark teal fill on hover */
        color: white;
        border: 1px solid #36a2eb;
    }

    /* Custom Section Header Styling for better separation */
    .sidebar-header {
        font-size: 18px;
        font-weight: 600;
        color: #87929f; /* Slightly lighter color for headers */
        margin-top: 15px;
        margin-bottom: 5px;
        padding-left: 15px;
    }

</style>
""", unsafe_allow_html=True)


# --- COURSE DATA MAPPING (Same as before) ---
course_skills_mapping_upgraded = {
    "Web Development": ["HTML5", "CSS3 (Sass/Less)", "JavaScript (ES6+)", "Responsive Design (Flexbox/Grid)", "Git/GitHub", "Frontend Frameworks (e.g., React, Vue, Angular)", "RESTful API Consumption"],
    # ... (Add the rest of your 40+ course entries here) ...
    # Note: I am truncating the list here to save space, but use the full list from the previous response.
    "Full Stack Web Development": ["MERN/MEAN/LAMP Stack Components", "Database Management (SQL/NoSQL)", "Security Best Practices (XSS, CSRF)", "Deployment (CI/CD, Heroku/Netlify)", "System Design Basics"]
}


# --- UPDATED SIDEBAR FUNCTION ---
def create_sidebar():
    """Renders the Streamlit sidebar with improved interactive navigation and system status."""
    st.sidebar.title("🎓 CourseRec AI")
    st.sidebar.markdown("Intelligent Learning Recommender")
    st.sidebar.markdown("---")

    # Navigation Section
    st.sidebar.markdown('<p class="sidebar-header">Navigation</p>', unsafe_allow_html=True)
    
    # Using Custom HTML/CSS links for compact and interactive layout
    # NOTE: These links currently don't change the Streamlit page; 
    # they are for visual effect only. You would use st.session_state 
    # in a real multi-page app to handle the page change.
    st.sidebar.markdown('<a href="#" class="sidebar-link">🏠 Home</a>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="sidebar-link">👤 User Profile</a>', unsafe_allow_html=True)

    # Core Tools Section
    st.sidebar.markdown('<p class="sidebar-header">Core Tools</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="sidebar-link">💡 Get Recommendations</a>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="sidebar-link">📚 Browse Courses</a>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="sidebar-link">⭐ Bookmarks</a>', unsafe_allow_html=True)


    # System Status Box (Same as before)
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Status")

    with st.sidebar.container(border=True):
        st.markdown("""
            **Courses:** 503
            
            **Skills Tracked:** 250
            
            **Engine:** **TF-IDF & Cosine Similarity**
        """)

# --- HOME PAGE CONTENT (Same as before) ---
def render_home_page():
    # ... (Your previous render_home_page and render_browse_courses functions remain here)
    # The home page functions are unchanged but are needed to make the app run.
    
    st.title("Course Recommendation AI")
    st.markdown("### Unleash your potential with **AI-powered course recommendations** tailored just for you.")
    st.markdown("##### Your next skill is just a click away.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 🧠 The AI Engine: Intelligent & Precise", unsafe_allow_html=True) 
        st.markdown("CourseRec AI uses advanced machine learning to transform your learning journey.")
        st.markdown("""
        * **Massive Data Scale:** Matches your profile against **500+ Courses** and **250 Skills** for superior accuracy.
        * **Semantic Matching:** Our engine uses **TF-IDF & Cosine Similarity** to understand the *meaning* behind your inputs.
        * **NEW: Personalized Insight:** Get a dedicated justification explaining *why* a course is the right match for you.
        """)

    with col2:
        st.markdown("## 🚀 Quick Start: Your Learning Roadmap", unsafe_allow_html=True)
        st.markdown("Follow these simple steps to begin your personalized learning adventure:")
        st.markdown("""
        1.  **Setup Your Profile:** Visit the **👤 User Profile** page to define your skills and interest domains.
        2.  **Get Recommendations:** Head to **💡 Get Recommendations**, select your **Difficulty Level**, and see tailored courses.
        3.  **Bridge the Gap:** Use the **AI Skill Gap Analysis** to find out exactly what you should learn next.
        """)
    st.markdown("---")
    
    render_browse_courses()

# --- BROWSE COURSES FUNCTIONALITY (Same as before) ---
def render_browse_courses():
    st.markdown("## 📚 Browse Courses & Skills", unsafe_allow_html=True)
    
    course_list = list(course_skills_mapping_upgraded.keys())
    
    selected_course = st.selectbox(
        "Select a Course/Domain to view required skills:",
        course_list,
        index=course_list.index("Web Development") if "Web Development" in course_list else 0
    )

    if selected_course:
        skills = course_skills_mapping_upgraded.get(selected_course, ["No specific skills found."])
        
        st.markdown(f"### Required Key Skills and Tools for: **{selected_course}**")
        
        num_skills = len(skills)
        num_rows = (num_skills + 2) // 3 
        
        skill_index = 0
        for _ in range(num_rows):
            cols = st.columns(3)
            for i in range(3):
                if skill_index < num_skills:
                    skill = skills[skill_index]
                    cols[i].markdown(f'<div class="skill-box">{skill}</div>', unsafe_allow_html=True)
                    skill_index += 1
                else:
                    cols[i].empty()
                
        if any(keyword in selected_course for keyword in ["Data", "Learning", "Intelligence", "Python"]):
            st.info("💡 **Note:** Strong foundation in **Statistics**, **Mathematics**, and **Problem-Solving** is crucial for this domain.")


# --- MAIN APPLICATION LOGIC ---
if __name__ == "__main__":
    create_sidebar()
    render_home_page()
