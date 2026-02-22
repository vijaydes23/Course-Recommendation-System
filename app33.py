import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random

# --- Constants: Upgraded Course Data from User Request ---

# 1. Detailed Skill Mapping (based on user input)
COURSE_SKILLS_MAPPING = {
    "Web Development": ["HTML5", "CSS3 (Sass/Less)", "JavaScript (ES6+)", "Responsive Design (Media Queries, Flexbox/Grid)", "Git/GitHub", "Frontend Frameworks (e.g., React, Vue, Angular)", "RESTful API Consumption"],
    "App Development": ["Java/Kotlin (Android) or Swift/SwiftUI (iOS)", "Flutter/React Native (Hybrid)", "API Integration (Retrofit, Alamofire)", "Mobile UI/UX Principles", "Firebase/AWS Mobile Hub", "State Management"],
    "Machine Learning": ["Python", "NumPy", "Pandas", "Scikit-learn", "TensorFlow/PyTorch", "Model Deployment (Flask/Streamlit)", "Statistics", "Linear Algebra", "Jupyter/Colab", "Model Evaluation Metrics"],
    "Data Science": ["Python", "R", "SQL (PostgreSQL/MySQL)", "Pandas", "NumPy", "Matplotlib/Seaborn", "Statistical Analysis", "Data Cleaning (ETL)", "Data Warehousing Basics", "Git/GitHub"],
    "Artificial intelligence": ["Python", "Deep Learning", "Neural Networks", "NLP (Natural Language Processing)", "Computer Vision (OpenCV)", "Algorithms", "Generative AI Basics", "Cloud Compute (GPU/TPU)"],
    "Cyber Security": ["Networking (TCP/IP)", "Linux (Command Line)", "Ethical Hacking Tools (e.g., Kali Linux, Wireshark)", "Penetration Testing", "Security Protocols (SSL/TLS)", "Firewalls/IDS/IPS", "Scripting (Bash/Python)"],
    "AR / VR": ["Unity or Unreal Engine", "C# (for Unity) or C++ (for Unreal)", "3D Modeling (Blender/Maya Basics)", "Spatial Computing", "Interaction Design", "SDKs (e.g., Oculus, ARKit)"],
    "UI & UX Designing": ["Figma/Sketch/Adobe XD", "Wireframing", "Prototyping", "Usability Testing", "Design Systems", "User Research", "Information Architecture", "Adobe Illustrator/Photoshop"],
    "Embedded Systems": ["C/C++", "Microcontrollers (Arduino/PIC/ARM)", "RTOS (Real-Time Operating Systems)", "Circuit Design (KiCad/Altium)", "Hardware Debugbing (Oscilloscope)"],
    "VLSI": ["Verilog/VHDL", "Digital Logic Design", "Circuit Simulation Tools (e.g., Spice)", "CMOS Technology", "ASIC/FPGA Design Flow", "Timing Analysis"],
    "Hybrid Electric Vehicle": ["Automotive Electronics (CAN Bus)", "Battery Technology (Lithium-ion)", "Power Electronics (Inverters/Converters)", "Vehicle Dynamics", "MATLAB/Simulink"],
    "Internet of Things & Robotics (Free Hardware Kit)": ["Python/C++", "Microcontroller Programming (e.g., Raspberry Pi, ESP32)", "Sensor Interfacing", "MQTT/CoAP Protocols", "Cloud Platforms (e.g., AWS IoT)", "Basic Soldering"],
    "Car Design": ["Sketching/Industrial Design", "CAD Software (e.g., CATIA, SolidWorks)", "Aerodynamics", "Material Science", "Feasibility Studies"],
    "IC Engine": ["Thermodynamics", "Combustion Principles", "Engine Performance Analysis", "CFD Simulation Basics (Ansys)", "Emission Control"],
    "CATIA": ["3D Modeling (Part/Assembly Design)", "Surface Modeling (Generative Shape Design)", "Drafting", "Engineering Drawing Standards", "Simulation Basics"],
    "Construction Planning and Management": ["MS Project/Primavera P6", "BIM (Building Information Modeling-Revit)", "Project Scheduling (CPM/PERT)", "Cost Estimation", "Risk Management"],
    "AutoCAD": ["2D Drafting", "3D Modeling Basics", "Technical Drawing Standards (ISO/ASME)"],
    "Nano Technology": ["Materials Science", "Quantum Mechanics Basics", "Chemistry", "Microscopy Techniques (SEM/TEM)", "Nanofabrication"],
    "Genetic Engineering": ["Molecular Biology Techniques (PCR, Gel Electrophoresis)", "Bioinformatics Basics (Sequence Analysis)", "Gene Editing (CRISPR)", "R Programming for Genomics"],
    "Molecular Biology": ["Cell Culture", "Protein Analysis (Western Blot)", "DNA/RNA Isolation", "Biochemistry", "Microscopy"],
    "Micro Biology": ["Sterile Techniques", "Microbial Culture", "Immunology Basics", "Staining Techniques"],
    "Finance": ["Financial Modeling (Excel)", "Valuation Techniques (DCF)", "Financial Statement Analysis", "Risk Management", "Bloomberg Terminal Basics (optional)"],
    "Digital Marketing": ["SEO/SEM (Google Ads, Bing)", "Content Marketing Strategy", "Google Analytics 4", "Email Marketing Tools (Mailchimp)", "A/B Testing", "Conversion Rate Optimization (CRO)"],
    "Stock Marketing": ["Technical Analysis (Charts, Indicators)", "Fundamental Analysis (Ratios)", "Trading Platforms (e.g., TradingView)", "Portfolio Management", "Risk Assessment"],
    "Marketing Management": ["Market Research", "Consumer Behavior", "Branding Strategy", "Pricing Models", "Marketing Mix (4 Ps)", "CRM Software Basics"],
    "Human Resource": ["HRIS Software (e.g., SAP HR/Workday)", "Recruitment Techniques (ATS)", "Labor Law Basics", "Performance Management Systems", "Employee Relations"],
    "Drone Engineering (Free Hardware Kit)": ["Aerodynamics", "Flight Control Systems (e.g., Pixhawk)", "RC Technology", "Basic Electronics", "Payload Integration", "Mission Planning Software"],
    "Microsoft Word": ["Document Formatting (Styles, Templates)", "Mail Merge", "Referencing and Citations (APA/MLA)", "Track Changes"],
    "Battery Management System ( BMS)": ["Power Electronics", "Li-ion Battery Characteristics", "Circuit Protection", "Microcontroller Programming", "State of Charge (SOC) Estimation"],
    "Supply Chain Management": ["Inventory Management Systems (ERP - SAP/Oracle)", "Logistics Optimization", "Forecasting Techniques", "Supplier Relationship Management (SRM)"],
    "Graphic Designing": ["Adobe Photoshop", "Adobe Illustrator", "InDesign (for print)", "Typography", "Color Theory", "Layout Design", "Vector Graphics"],
    "Metaverse": ["Unity/Unreal Engine", "Blockchain Basics", "3D Asset Creation (Modeling/Texturing)", "Virtual Interaction Design", "WebXR"],
    "Social Media Marketing": ["Platform Algorithms (e.g., Instagram, Facebook, TikTok)", "Content Scheduling Tools (Buffer/Hootsuite)", "Audience Targeting", "Analytics", "Paid Ad Campaigns"],
    "Investment Banking": ["Advanced Excel & Financial Modeling", "M&A (Mergers & Acquisitions) Valuation", "Pitchbook Creation", "Capital Markets"],
    "Bitcoin / Blockchain": ["Cryptography", "Solidity (for Ethereum)", "Decentralized Applications (DApps)", "Smart Contracts", "Web3.js/Ethers.js"],
    "Data Analytics": ["SQL (Advanced Queries)", "PowerBI/Tableau/Looker Studio", "Excel (Advanced)", "Statistical Software (e.g., SPSS/SAS)", "Data Cleansing", "Data Storytelling"],
    "WEB 3 Web Development": ["Solidity", "JavaScript/TypeScript", "React/Next.js", "Truffle/Hardhat (Development Frameworks)", "Metamask Integration", "IPFS"],
    "Entrepreneurship": ["Business Plan Writing", "Financial Projections", "Market Validation", "Pitching/Presentation Skills", "Lean Startup Methodology", "Fundraising Basics"],
    "Python with Data Science ( IBM )": ["Python (Advanced)", "Pandas", "NumPy", "Jupyter Notebooks", "Data Wrangling", "Basic ML Models"],
    "Data Visualization with R ( IBM )": ["R", "ggplot2", "Shiny (for Web Apps)", "Statistical Graphics", "Data Munging (dplyr)"],
    "Cloud Computing": ["Virtualization", "Networking (Subnets, Load Balancers)", "Containers (Docker/Kubernetes)", "Automation Scripts (e.g., Terraform, Ansible)", "Security Principles"],
    "Business Analytics": ["SQL", "Excel (Pivot Tables, VLOOKUP)", "Statistical Modeling", "Data Storytelling", "A/B Testing", "Business Intelligence (BI)"],
    "Amazon Web Service (AWS)": ["EC2", "S3", "Lambda (Serverless)", "VPC", "IAM", "Cloud Architecture (Well-Architected Framework)", "CloudWatch"],
    "C,C++ with DSA": ["C/C++ Syntax", "Pointers/References", "Linked Lists, Trees, Graphs", "Time and Space Complexity Analysis", "Algorithmic Problem Solving (LeetCode/HackerRank)"],
    "Java": ["OOP (Object-Oriented Programming)", "Collections Framework", "Multithreading", "Spring/Hibernate Basics", "JVM Internals"],
    "Mern Stack": ["MongoDB (NoSQL)", "Express.js", "React (Hooks, Redux/Context API)", "Node.js (Async/Await)", "REST APIs", "Authentication (JWT)", "Testing (Jest/Mocha)"],
    "Python": ["Syntax and Data Structures (Lists, Dicts)", "Functions and Modules", "Object-Oriented Python", "Debugging", "Pip/Virtual Environments"],
    "Full Stack Web Development": ["MERN/MEAN/LAMP Stack Components", "Database Management (SQL/NoSQL)", "Security Best Practices (XSS, CSRF)", "Deployment (CI/CD, Heroku/Netlify)", "System Design Basics"]
}

INTEREST_DOMAINS = list(COURSE_SKILLS_MAPPING.keys())

# Generate a master list of all unique, cleaned skills
ALL_TECHNICAL_SKILLS = set()
for skills_list in COURSE_SKILLS_MAPPING.values():
    for skill in skills_list:
        # Clean skill name (remove text in parentheses) for better matching
        cleaned_skill = re.sub(r'\s*\(.*\)', '', skill).strip()
        ALL_TECHNICAL_SKILLS.add(cleaned_skill)
ALL_TECHNICAL_SKILLS = sorted(list(ALL_TECHNICAL_SKILLS))

# --- Data Generation Function (Updated to use new mapping) ---

def generate_course_data():
    courses = []
    course_id = 1
    platforms = ['Udemy', 'Coursera', 'edX', 'LinkedIn Learning', 'Pluralsight']
    difficulties = ['Beginner', 'Intermediate', 'Advanced']
    prices = ['Free', '₹499', '₹799', '₹1199', '₹1999', '₹2499']
    
    TITLE_TEMPLATES = [
        "Mastering {skill} in {domain}", 
        "The Complete {domain} Bootcamp: From Zero to Expert",
        "Professional Certification in {skill1} and {skill2}",
        "Advanced {domain} Techniques with {skill}",
    ]

    for domain, skills_list in COURSE_SKILLS_MAPPING.items():
        base_skills = [re.sub(r'\s*\(.*\)', '', s).strip() for s in skills_list]
        
        # Ensure domain has enough skills to pick from
        if len(base_skills) < 2: continue
        
        # Generate 8 to 12 courses per domain for robust data
        count = random.randint(8, 12) 
        
        for i in range(count): 
            # Select 4-6 primary skills for the course
            k_skills = min(len(base_skills), random.randint(4, 6))
            primary_skills = random.sample(base_skills, k=k_skills)
            
            skill1, skill2 = primary_skills[0], primary_skills[1]
            title_template = random.choice(TITLE_TEMPLATES)
            
            # Smart title generation
            if "{skill1}" in title_template:
                full_title = title_template.format(skill1=skill1, skill2=skill2)
            else:
                full_title = title_template.format(domain=domain, skill=skill1)
                
            # Add a unique identifier and format for better readability
            full_title = f"{full_title.replace('Mastering', 'Master')} ({domain} Focus v{i+1})"
            
            # Generate a description focusing on key skills
            description = (
                f"A comprehensive course focusing on **{skill1}** and **{skill2}** basics, "
                f"leading to advanced topics in {domain}. Key tools include "
                f"{', '.join(primary_skills[2:])}. Includes hands-on projects and certification prep."
            )

            courses.append({
                'course_id': course_id,
                'title': full_title,
                'description': description,
                'category': domain,
                'platform': random.choice(platforms),
                'difficulty': random.choice(difficulties),
                'rating': round(random.uniform(4.0, 4.9), 1),
                'duration': random.randint(10, 80), 
                # Skills are stored as a space-separated string for ML model
                'skills': " ".join(primary_skills),
                'enroll_link': f"https://example.com/course/{course_id}", 
                'price': random.choice(prices)
            })
            course_id += 1
    
    random.shuffle(courses)
    return courses

# --- Custom CSS (Unchanged Aesthetics) ---
CUSTOM_CSS = """
<style>
/* Define a more cohesive and vibrant color palette */
:root {
    --vibrant-blue: #00A3FF;       /* Primary interactive color */
    --accent-green: #28a745;       /* For success/completion */
    --accent-orange: #fd7e14;      /* For warnings/highlights */
    --primary-color-rgb: 0, 163, 255; /* RGB for shadows */
    --dark-bg-color: #0d1117;      /* Dark theme background */
    --dark-secondary-bg: #161b22;  /* Dark theme secondary bg */
    --light-text-color: #c9d1d9;   /* Light text for dark theme */
}

/* Ensure Streamlit's default dark theme aligns with custom styles */
[data-theme="dark"] {
    --background-color: var(--dark-bg-color);
    --secondary-background-color: var(--dark-secondary-bg);
    --text-color: var(--light-text-color);
    --primary-color: var(--vibrant-blue); 
}

/* General Styling */
.main { padding-top: 30px; }

/* ------------------ Sidebar Navigation Styling ------------------ */
[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color); 
    box-shadow: 2px 0 10px rgba(0,0,0,0.15);
    padding-top: 20px;
}

/* Navbar Buttons in Sidebar - Polished */
[data-testid="stSidebar"] .stButton>button {
    background-color: var(--secondary-background-color); 
    color: var(--text-color);
    border-radius: 12px; 
    margin-bottom: 10px; 
    font-weight: 600;
    font-size: 1.05rem; 
    padding: 12px 18px; 
    width: 100%;
    text-align: left;
    transition: background-color 0.3s, transform 0.2s, color 0.3s, box-shadow 0.3s;
    border: 1px solid rgba(var(--primary-color-rgb), 0.3); 
}
[data-testid="stSidebar"] .stButton>button:hover {
    background-color: var(--vibrant-blue); 
    color: white !important; 
    transform: translateY(-2px); 
    box-shadow: 0 5px 15px rgba(var(--primary-color-rgb), 0.5); 
    border-color: var(--vibrant-blue);
}

/* Sidebar Title and Subtitle */
[data-testid="stSidebar"] h2 {
    color: var(--vibrant-blue);
    text-align: center;
    margin-bottom: 5px;
}
[data-testid="stSidebar"] h3 {
    color: var(--text-color);
    text-align: center;
    font-size: 1.1rem;
    font-weight: normal;
    margin-bottom: 20px;
}
[data-testid="stSidebar"] h4 {
    color: var(--vibrant-blue);
    margin-top: 20px;
    margin-bottom: 10px;
}

/* ------------------ Course Card Styling (Animated Hover) ------------------ */
.course-card {
    padding: 25px;
    margin-bottom: 25px;
    border-radius: 15px; 
    border-left: 6px solid var(--vibrant-blue); 
    background-color: var(--secondary-background-color);
    box-shadow: 0 6px 15px rgba(0,0,0,0.15); 
    transition: box-shadow 0.3s ease-in-out, transform 0.3s;
    cursor: pointer;
}
.course-card:hover {
    box-shadow: 0 12px 28px rgba(var(--primary-color-rgb), 0.4); 
    transform: translateY(-7px); 
}

/* Skill Tags (Adjusted for better contrast) */
.skill-tag {
    display: inline-block;
    background-color: var(--secondary-background-color);
    color: var(--vibrant-blue); 
    padding: 6px 14px; 
    margin: 4px 3px;
    border-radius: 25px; 
    font-size: 0.9rem; 
    font-weight: 500;
    border: 1px solid var(--vibrant-blue);
    transition: background-color 0.2s, color 0.2s;
}
.skill-tag:hover {
    background-color: var(--vibrant-blue);
    color: white;
}

/* Highlighted Next Skill Tag (New Feature Styling) */
.next-skill-tag {
    display: inline-block;
    background-color: var(--accent-orange); 
    color: white; 
    padding: 6px 14px; 
    margin: 4px 3px;
    border-radius: 25px; 
    font-size: 0.9rem; 
    font-weight: 700;
    border: 1px solid var(--accent-orange);
}

/* Primary Button Styling (Enroll Now) - Vibrant and Animated */
.primary-btn {
    background-color: var(--vibrant-blue); 
    color: white;
    padding: 12px 15px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    width: 100%;
    font-weight: bold;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    transition: background-color 0.3s, transform 0.2s, box-shadow 0.3s;
    margin-top: 10px; 
}
.primary-btn:hover {
    background-color: #0077c2;
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(var(--primary-color-rgb), 0.6);
}

/* Secondary Buttons (Bookmark, Edit, Logout, Mark as Completed) */
.stButton>button[kind="secondary"] { 
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    border: 1px solid var(--vibrant-blue);
    border-radius: 10px;
    padding: 10px 15px;
    margin-top: 10px;
    width: 100%;
    transition: background-color 0.3s, color 0.3s, transform 0.2s;
}
.stButton>button[kind="secondary"]:hover {
    background-color: var(--vibrant-blue);
    color: white;
    transform: scale(1.01);
}

/* Specific styling for 'Mark as Completed' button */
.stButton>button[key^="complete_"] { 
    background-color: var(--accent-green);
    color: white;
    border: none;
}
.stButton>button[key^="complete_"]:hover {
    background-color: #218838; 
    color: white;
}


/* Home Page Info Boxes */
.info-box {
    padding: 25px;
    border-radius: 12px;
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(var(--primary-color-rgb), 0.3); 
    margin-bottom: 25px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.info-box h4 {
    color: var(--vibrant-blue);
    margin-top: 0;
}

/* Custom Metric Styling: Centered and Animated */
[data-testid="stMetric"] {
    background-color: var(--secondary-background-color); 
    border: 1px solid var(--vibrant-blue);
    border-radius: 12px;
    display: flex; 
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 15px; 
    transition: box-shadow 0.3s, transform 0.3s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 15px rgba(var(--primary-color-rgb), 0.3);
    transform: translateY(-3px);
}
[data-testid="stMetric"] div[data-testid="stMarkdownContainer"] p {
    color: var(--vibrant-blue) !important;
    font-weight: bold;
    font-size: 1rem; 
}
[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--text-color); 
    font-size: 2.2rem; 
    font-weight: 600;
}


/* Profile Card Styling for View Mode */
.profile-card {
    padding: 30px;
    border-radius: 15px;
    border: 1px solid rgba(var(--primary-color-rgb), 0.3);
    background-color: var(--secondary-background-color);
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    margin-bottom: 30px; 
}
.profile-field-label {
    font-weight: bold;
    color: var(--vibrant-blue);
    font-size: 1.1rem; 
    margin-top: 15px;
    margin-bottom: 5px;
}
.profile-field-value {
    margin-bottom: 15px; 
    font-size: 1.1rem;
    color: var(--text-color);
}
.profile-skill-tag { 
    background-color: var(--vibrant-blue); 
    color: white; 
    padding: 5px 12px;
    margin: 4px 2px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid var(--vibrant-blue);
    display: inline-block;
}

/* Header styling */
h1 {
    color: var(--vibrant-blue);
}
h3 {
    color: var(--vibrant-blue);
}
</style>
"""

# --- Recommender Class (ML Logic) ---
class CourseRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self):
        self.df = pd.DataFrame(generate_course_data())
        self.df = self.df.fillna({'description': '', 'category': 'Unknown', 'skills': '', 'difficulty': 'Beginner', 'rating': 4.0})
        
    def preprocess_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        # Clean up text by keeping alphanumeric and spaces only
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
        return ' '.join(text.split())
    
    def preprocess_data(self):
        self.df['title_clean'] = self.df['title'].apply(self.preprocess_text)
        self.df['description_clean'] = self.df['description'].apply(self.preprocess_text)
        self.df['category_clean'] = self.df['category'].apply(self.preprocess_text)
        self.df['skills_clean'] = self.df['skills'].apply(self.preprocess_text)
        
        # Combine features for TF-IDF Vectorization
        self.df['combined_features'] = (
            self.df['title_clean'] + ' ' +
            self.df['description_clean'] + ' ' +
            self.df['category_clean'] * 5 + ' ' + # Weighting category higher
            self.df['skills_clean'] * 3          # Weighting skills higher
        )
        
    def build_model(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000, # Increased max_features for more detailed skills
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
        
    def get_recommendations(self, user_skills, interest_domains, difficulty_filter, top_n=10):
        if self.df is None or self.tfidf_matrix is None: return []
        
        # Create user profile string, weighting interest domains highly
        user_profile = ' '.join(user_skills) + ' ' + ' '.join(interest_domains) * 8
        user_profile_clean = self.preprocess_text(user_profile)
        
        temp_df = self.df.copy()
        
        if difficulty_filter != "All":
            temp_df = temp_df[temp_df['difficulty'] == difficulty_filter]
            if temp_df.empty: return []

        if user_profile_clean:
            # Re-initialize vectorizer for temporary filtered data
            temp_vectorizer = TfidfVectorizer(
                max_features=2000, stop_words='english', ngram_range=(1, 2)
            )
            temp_tfidf_matrix = temp_vectorizer.fit_transform(temp_df['combined_features'])
            user_tfidf = temp_vectorizer.transform([user_profile_clean])
            
            cosine_sim = cosine_similarity(user_tfidf, temp_tfidf_matrix).flatten()
            similarity_scores = list(enumerate(cosine_sim))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        else:
            # Fallback to rating-based sort if profile is empty
            similarity_scores = [(i, row['rating'] / 5.0) for i, row in temp_df.iterrows()]
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        completed_courses = st.session_state.user_profile['completed_courses']
        
        # Get the original index of the courses in the filtered/sorted list
        original_indices = temp_df.iloc[[idx for idx, score in similarity_scores]].index.tolist()
        
        for i, (idx, score) in enumerate(similarity_scores):
            original_idx = original_indices[i] 
            course_id = self.df.loc[original_idx]['course_id']
            
            if course_id not in completed_courses:
                course_data = self.df.loc[original_idx].to_dict()
                course_data['similarity_score'] = float(score)
                recommendations.append(course_data)
            
            if len(recommendations) >= top_n: break
        
        return recommendations

# --- AI Helper Function: Personalized Justification (NEW FEATURE 2) ---

def generate_justification(course_data, user_profile):
    user_skills = set([re.sub(r'\s*\(.*\)', '', s).strip().lower() for s in user_profile['skills']])
    course_skills = set([s.lower() for s in course_data['skills'].split()])
    
    overlap = user_skills.intersection(course_skills)
    missing = course_skills.difference(user_skills)
    
    justification_points = []
    
    # Point 1: Match based on Interest Domain
    if course_data['category'] in user_profile['interest_domains']:
        justification_points.append(f"**🎯 Career Alignment:** This course is a direct match for your chosen domain, **{course_data['category']}**.")
    else:
        justification_points.append(f"**💡 Foundational Learning:** Recommended due to its strong core topics in **{course_data['category']}**.")

    # Point 2: Skill Overlap (Leveraging existing skills)
    if overlap:
        overlap_list = [s for s in course_data['skills'].split() if s.lower() in overlap][:2]
        justification_points.append(f"**✅ Leverage Existing Skills:** You can build on your background in **{', '.join(overlap_list)}** for a faster learning curve.")
        
    # Point 3: Gap Filling (Learning new skills)
    if missing:
        missing_list = [s for s in course_data['skills'].split() if s.lower() in missing][:2]
        justification_points.append(f"**⭐ Critical Skill Acquisition:** Master **{', '.join(missing_list)}**, which are essential skills required by this domain.")
    else:
        justification_points.append("**🚀 Advanced Mastery:** Perfect for solidifying your expertise and moving towards specialized, advanced concepts.")
        
    return justification_points

# --- AI Helper Function: Study Plan Generation ---
def generate_study_plan(course_data):
    title = course_data['title']
    skills = course_data['skills'].split()
    duration = course_data['duration']
    
    # 1. Core Modules (Simulated)
    modules = [
        f"Module 1: Foundational Setup and {random.choice(skills[:2])} Principles",
        f"Module 2: Core Concepts in {course_data['category']} and Advanced Data Handling",
        f"Module 3: Practical Implementation using {random.choice(skills[2:4])} and hands-on projects",
        f"Module 4: Advanced Techniques, Optimization, and Industry Best Practices",
        f"Module 5: Final Capstone Project, Portfolio Building, and Certification Prep"
    ]
    
    # 2. Core Tools
    core_tools = [s for s in skills if len(s) > 3 and s[0].isupper()][:5]
    
    # 3. Estimated Time
    time_estimate = f"Total Estimated Time: {duration} hours. Recommended Pace: 5-8 hours/week for completion."

    return {
        'modules': modules,
        'tools': core_tools,
        'time': time_estimate,
        'description': course_data['description']
    }

# --- AI Helper Function: Next Logical Skill (NEW FEATURE 1) ---

def get_next_logical_skill(user_skills, course_skills):
    """
    Finds the most relevant skill in the course that the user doesn't have.
    """
    user_skills_lower = set([re.sub(r'\s*\(.*\)', '', s).strip().lower() for s in user_skills])
    course_skills_list = [re.sub(r'\s*\(.*\)', '', s).strip() for s in course_skills.split()]
    
    # Look for 1-2 core skills the user is missing
    missing_core_skills = [
        s for s in course_skills_list 
        if s.lower() not in user_skills_lower
    ]
    
    if missing_core_skills:
        # Prioritize the skills that appear first in the course's skill list (often the most foundational)
        return missing_core_skills[:2] 
        
    return None

# --- AI Helper Function (Skill Gap Analysis) ---

def ai_skill_gap_analysis(df, user_skills, interest_domains):
    if not interest_domains:
        return None, None
    
    target_domains = [d for d in interest_domains if d in df['category'].unique()]
    
    if not target_domains: return None, None

    required_skills_list = []
    
    for domain in target_domains:
        # Use the master skill list for the domain for more accurate gap analysis
        required_skills_list.extend(COURSE_SKILLS_MAPPING.get(domain, [])) 
    
    if not required_skills_list: return None, None

    skill_counts = pd.Series(required_skills_list).value_counts()
    
    user_skills_clean = set([re.sub(r'\s*\(.*\)', '', s).strip() for s in user_skills])
    
    # Find skills the user doesn't have, case-insensitively
    gap_skills = skill_counts[~skill_counts.index.map(lambda x: re.sub(r'\s*\(.*\)', '', x).strip()).str.lower().isin([s.lower() for s in user_skills_clean])]
    
    top_gap_skills = gap_skills.head(5).index.tolist()

    if not top_gap_skills: return None, None
    
    target_skill = top_gap_skills[0] 
    
    # Find a foundational course for the top gap skill
    foundational_course = df[
        (df['difficulty'] == 'Beginner') & 
        (df['skills'].str.contains(re.escape(target_skill), case=False, na=False))
    ].sort_values(by='rating', ascending=False).head(1)
    
    return top_gap_skills, foundational_course.to_dict('records')[0] if not foundational_course.empty else None

# --- Streamlit UI Functions ---

def initialize_session_state():
    """Initialize session state and model."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True) 

    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'roll_no': '',
            'department': '',
            'skills': [],
            'interest_domains': [],
            'completed_courses': [],
            'bookmarks': [],
            'edit_mode': True 
        }
    
    if not st.session_state.user_profile.get('name') or not st.session_state.user_profile.get('roll_no'):
        st.session_state.user_profile['edit_mode'] = True
    elif 'edit_mode' not in st.session_state.user_profile:
        st.session_state.user_profile['edit_mode'] = False
    
    if 'recommender' not in st.session_state:
        st.session_state.recommender = CourseRecommender()
        # This will now use the new, detailed COURSE_SKILLS_MAPPING
        st.session_state.recommender.load_data() 
        st.session_state.recommender.preprocess_data()
        st.session_state.recommender.build_model()
    
    if 'last_recommendations' not in st.session_state:
        st.session_state.last_recommendations = []
    
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

def display_course_card(course, show_similarity=False, show_completion_button=False):
    """
    Display a course card with the added features: Next Logical Skill and Justification.
    """
    profile = st.session_state.user_profile

    # Generate NEW FEATURE 2: Personalized Justification
    justification_points = generate_justification(course, profile)
    
    # Generate NEW FEATURE 1: Next Logical Skill
    next_skills = get_next_logical_skill(profile['skills'], course['skills'])

    st.markdown('<div class="course-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown(f"**<h3 style='margin-top: 0; color: var(--vibrant-blue);'>{course['title']}</h3>**", unsafe_allow_html=True)
        st.markdown(f"**Platform:** `{course['platform']}` | **Category:** `{course['category']}` | **Difficulty:** `{course['difficulty']}`")
        
        st.markdown(f"**Rating:** ⭐ {course['rating']} | **Duration:** {course['duration']} hours | **Price:** **{course['price']}**")
        
        # Display Next Logical Skill (NEW FEATURE 1)
        if next_skills:
            next_skill_chips = "".join([f'<span class="next-skill-tag">🔥 {skill}</span>' for skill in next_skills])
            st.markdown(f"**Target Skills to Master:** {next_skill_chips}", unsafe_allow_html=True)
        
        skills = course['skills'].split()
        display_skills = skills[:6]
        skill_chips = "".join([f'<span class="skill-tag">{skill}</span>' for skill in display_skills])
        st.markdown(f"**Course Skills:** {skill_chips}", unsafe_allow_html=True)
        
        st.markdown(f"<p style='font-style: italic;'>{course['description']}</p>", unsafe_allow_html=True)
    
    with col2:
        if show_similarity and 'similarity_score' in course:
            similarity_percent = course['similarity_score'] * 100
            st.metric("Match Score", f"{similarity_percent:.1f}%")
        
        bookmark_key = f"bookmark_{course['course_id']}"
        is_bookmarked = course['course_id'] in st.session_state.user_profile['bookmarks']
        
        bookmark_label = "★ Bookmarked" if is_bookmarked else "⭐ Bookmark"
        
        if st.button(bookmark_label, key=bookmark_key, use_container_width=True, type="secondary"):
            if not is_bookmarked:
                st.session_state.user_profile['bookmarks'].append(course['course_id'])
                st.toast(f"✅ Bookmarked: {course['title']}", icon="⭐")
            else:
                st.session_state.user_profile['bookmarks'].remove(course['course_id'])
                st.toast(f"🗑️ Removed Bookmark: {course['title']}", icon="📝")
            st.rerun() 
            
        st.markdown(
            f"""
            <a href="{course['enroll_link']}" target="_blank">
                <button class="primary-btn">
                    🎯 Enroll Now
                </button>
            </a>
            """, unsafe_allow_html=True
        )
        
        if show_completion_button and course['course_id'] not in st.session_state.user_profile['completed_courses']:
            if st.button("✅ Mark as Completed", key=f"complete_{course['course_id']}", use_container_width=True):
                st.session_state.user_profile['completed_courses'].append(course['course_id'])
                if course['course_id'] in st.session_state.user_profile['bookmarks']:
                    st.session_state.user_profile['bookmarks'].remove(course['course_id'])
                st.toast(f"🥳 Completed: {course['title']}! Removed from bookmarks.", icon="✅")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Course Details Expander (The Study Material Pop-up) ---
    with st.expander(f"📚 View AI Study Plan & Justification for {course['title']}", expanded=False):
        
        st.markdown("#### ⭐ Personalized Justification (Why this course is for you)")
        st.markdown('<div class="info-box" style="border-left: 5px solid var(--accent-green);">', unsafe_allow_html=True)
        for point in justification_points:
            st.markdown(f"- {point}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        study_plan_data = generate_study_plan(course)
        st.markdown(f"#### 🧠 AI-Generated Study Plan & Tools (Duration: {study_plan_data['time']})")
        
        col_plan, col_tools = st.columns(2)
        
        with col_plan:
            st.markdown("##### Structured Modules")
            for module in study_plan_data['modules']:
                st.markdown(f"- {module}")
        
        with col_tools:
            st.markdown("##### Core Tools to Master")
            tool_chips = "".join([f'<span class="profile-skill-tag">{t}</span>' for t in study_plan_data['tools']])
            st.markdown(tool_chips, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"[Go to Course Enrollment Page]({course['enroll_link']})")


# --- Page Definitions ---

def show_home_page():
    st.markdown("<h1 style='text-align: center;'>🎓 CourseRec AI: Your Personalized Learning Path</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.image("https://www.streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.svg", width=150) 
    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <p style='font-size: 1.3rem; color: var(--text-color);'>
                Unleash your potential with **AI-powered course recommendations** tailored just for you.
            </p>
            <p style='font-size: 1.1rem; color: var(--vibrant-blue); font-weight: bold;'>
                Your next skill is just a click away.
            </p>
        </div>
        """, unsafe_allow_html=True)

    col_ai_info, col_setup_info = st.columns(2)
    
    with col_ai_info:
        st.markdown("<h3 style='color: var(--vibrant-blue); text-align: center;'>🧠 The AI Engine: Intelligent & Precise</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <p style='font-size: 1.05rem;'>CourseRec AI uses advanced machine learning to transform your learning journey. We dive deep into **400+ courses** to find the perfect fit.</p>
            <ul style='padding-left: 20px; margin-top: 15px;'>
                <li><span style='font-weight: bold;'>Massive Data Scale:</span> Matching against **{total_skills} Skills** and {total_courses} courses for superior accuracy.</li>
                <li><span style='font-weight: bold;'>Semantic Matching:</span> Our engine uses TF-IDF and Cosine Similarity to understand the *meaning* behind your inputs.</li>
                <li><span style='font-weight: bold;'>NEW: Personalized Insight:</span> Get a dedicated justification explaining *why* a course is the right match.</li>
            </ul>
        </div>
        """.format(total_skills=len(ALL_TECHNICAL_SKILLS), total_courses=len(st.session_state.recommender.df)), unsafe_allow_html=True)

    with col_setup_info:
        st.markdown("<h3 style='color: var(--vibrant-blue); text-align: center;'>🚀 Quick Start: Your Learning Roadmap</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="border-left: 5px solid var(--accent-orange);">
            <p style='font-size: 1.05rem;'>Follow these simple steps to begin your personalized learning adventure:</p>
            <ol style='padding-left: 20px; margin-top: 15px;'>
                <li><span style='font-weight: bold;'>Setup Your Profile:</span> Visit the **👤 User Profile** page to define your skills, department, and **Interest Domains**.</li>
                <li><span style='font-weight: bold;'>Get Recommendations:</span> Head to **💡 Get Recommendations**, choose your desired **Difficulty Level**, and see tailored courses.</li>
                <li><span style='font-weight: bold;'>NEW: Bridge the Gap:</span> Use the **AI Skill Gap Analysis** to find out what you should learn next.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: var(--text-color);'>📊 System Overview</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    df = st.session_state.recommender.df
    with col1:
        st.metric("Total Courses", len(df), help="The total number of courses available in the dataset.")
    with col2:
        st.metric("Unique Skills", len(ALL_TECHNICAL_SKILLS), help="The number of unique skills tracked in the system.")
    with col3:
        st.metric("Completed Courses", len(st.session_state.user_profile['completed_courses']), help="Courses you have marked as complete.")
    with col4:
        avg_rating = df['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐", help="Average rating across all available courses.")


def show_profile_page():
    st.title("👤 User Profile Management")
    st.markdown("---")
    
    profile = st.session_state.user_profile
    is_profile_set = profile.get('name') and profile.get('roll_no') and profile.get('department')
    
    if is_profile_set and not profile.get('edit_mode'):
        # --- View Profile Mode ---
        st.subheader("Your Current Learning Profile")
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        
        col_name_roll, col_dept = st.columns(2)
        
        with col_name_roll:
            st.markdown('<p class="profile-field-label">Full Name</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="profile-field-value">{profile.get("name", "N/A")}</p>', unsafe_allow_html=True)
            st.markdown('<p class="profile-field-label">Roll No. / Student ID</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="profile-field-value">{profile.get("roll_no", "N/A")}</p>', unsafe_allow_html=True)
        
        with col_dept:
            st.markdown('<p class="profile-field-label">Department / Major</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="profile-field-value">{profile.get("department", "N/A")}</p>', unsafe_allow_html=True)

        st.markdown('<p class="profile-field-label">Current Skills</p>', unsafe_allow_html=True)
        if profile['skills']:
            skill_chips = "".join([f'<span class="profile-skill-tag">{s}</span>' for s in profile['skills']])
            st.markdown(f'<p class="profile-field-value">{skill_chips}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="profile-field-value" style="font-style: italic; color: grey;">No skills added yet.</p>', unsafe_allow_html=True)

        st.markdown('<p class="profile-field-label">Interest Domain (Career Goal)</p>', unsafe_allow_html=True)
        if profile['interest_domains']:
            domain_chips = "".join([f'<span class="profile-skill-tag">{d}</span>' for d in profile['interest_domains']])
            st.markdown(f'<p class="profile-field-value">{domain_chips}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="profile-field-value" style="font-style: italic; color: grey;">No interest domains selected yet.</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---") 
        
        # Action buttons for view mode, placed at the bottom
        col_edit, col_logout = st.columns(2)
        with col_edit:
            if st.button("✏️ Edit Profile", use_container_width=True, type="primary", key="view_edit_btn"):
                st.session_state.user_profile['edit_mode'] = True
                st.rerun()
        with col_logout:
            if st.button("🚪 Logout / Switch User", use_container_width=True, type="secondary", key="view_logout_btn"):
                st.session_state.user_profile = {
                    'name': '', 'roll_no': '', 'department': '', 'skills': [], 
                    'interest_domains': [], 'completed_courses': [], 'bookmarks': [], 'edit_mode': True
                }
                st.toast("Profile cleared. Please enter new user details.", icon="👋")
                st.rerun()
                
    else:
        # --- Edit Profile Mode ---
        st.subheader("Enter or Update Your Details")
        st.info("💡 Complete your profile to unlock personalized course recommendations.")
        
        with st.form("user_profile_form"):
            st.markdown("#### Personal & Academic Details")
            
            col_edit1, col_edit2, col_edit3 = st.columns(3)
            with col_edit1:
                name = st.text_input("**Full Name**", value=profile.get('name', ''), placeholder="John Doe")
            with col_edit2:
                roll_no = st.text_input("**Roll No. / Student ID**", value=profile.get('roll_no', ''), placeholder="2024CS001")
            with col_edit3:
                department = st.text_input("**Department / Major**", value=profile.get('department', ''), placeholder="Computer Science")
            
            st.markdown("#### Current Skills & Career Goals")
            
            col_edit4, col_edit5 = st.columns(2)
            
            with col_edit4:
                skills = st.multiselect(
                    "**Current Skills**",
                    options=ALL_TECHNICAL_SKILLS,
                    default=profile['skills'],
                    help=f"Select skills you already possess (Total: {len(ALL_TECHNICAL_SKILLS)})."
                )
            
            with col_edit5:
                interest_domains = st.multiselect(
                    "**Interest Domain (Career Goal)**",
                    options=INTEREST_DOMAINS,
                    default=profile['interest_domains'],
                    help="The area you want to specialize in (Max 3 for best results)."
                )
            
            # --- Submit Button for the Edit Form ---
            if st.form_submit_button("💾 Save Profile & Finish Editing", use_container_width=True, type="primary"):
                if not name or not roll_no or not department:
                    st.error("Please enter your Full Name, Roll No., and Department.")
                else:
                    st.session_state.user_profile.update({
                        'name': name, 'roll_no': roll_no, 'department': department, 
                        'skills': skills, 'interest_domains': interest_domains,
                        'edit_mode': False 
                    })
                    st.success("🎉 Profile updated and saved! Redirecting to view mode...")
                    st.rerun()

    st.markdown("---")
    st.subheader("✅ Course Completion Tracker")
    st.info(f"You have **{len(profile['completed_courses'])}** completed courses. These are excluded from recommendations.")
    
    if profile['completed_courses']:
        if st.button("🗑️ Clear All Completed Courses", key="clear_completed", type="secondary"):
            st.session_state.user_profile['completed_courses'] = []
            st.toast("Cleared all completed courses.", icon="🧹")
            st.rerun()


def show_recommendations_page():
    st.title("💡 AI-Powered Course Recommendations")
    st.markdown("---")
    
    profile = st.session_state.user_profile
    if not profile['skills'] and not profile['interest_domains']:
        st.warning("⚠️ **Please complete your profile first!** Add your skills and **Interest Domains** to get personalized recommendations.")
        return
    
    st.subheader("⚙️ Recommendation Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        num_recommendations = st.slider(
            "**Number of Courses**", 
            min_value=3, max_value=15, value=10,
            help="How many top-matching courses would you like to see?"
        )
    with col2:
        difficulty_filter = st.selectbox(
            "**Desired Difficulty Level**",
            options=["All"] + ['Beginner', 'Intermediate', 'Advanced'],
            help="Filter recommendations by difficulty."
        )

    if st.button("🎯 Get AI Recommendations", use_container_width=True, type="primary"):
        with st.spinner("🔍 Analyzing your profile and finding the best courses..."):
            
            try:
                recommendations = st.session_state.recommender.get_recommendations(
                    user_skills=profile['skills'],
                    interest_domains=profile['interest_domains'],
                    difficulty_filter=difficulty_filter,
                    top_n=num_recommendations
                )
                
                st.session_state.last_recommendations = recommendations
                st.success(f"✅ Found {len(recommendations)} highly relevant courses!")
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
    
    # --- AI Skill Gap Analysis ---
    st.markdown("---")
    st.subheader("🤖 AI Skill Gap Analysis (Next Step Suggestion)")
    
    if profile['interest_domains']:
        with st.spinner("Analyzing skill deficiencies in your target domain..."):
            gap_skills, foundational_course = ai_skill_gap_analysis(
                st.session_state.recommender.df,
                profile['skills'],
                profile['interest_domains']
            )

        if gap_skills:
            st.markdown(f"**Your Target Domain(s):** <span style='color: var(--vibrant-blue); font-weight: bold;'>{', '.join(profile['interest_domains'])}</span>", unsafe_allow_html=True)
            st.warning("🚨 **Potential Skill Gaps Detected!** You are missing these high-demand foundational skills:")
            
            gap_chips = "".join([f'<span class="skill-tag" style="background-color: var(--accent-orange); color: white; border-color: var(--accent-orange);">{s}</span>' for s in gap_skills])
            st.markdown(gap_chips, unsafe_allow_html=True)
            
            if foundational_course:
                st.info(f"💡 **Recommended Next Step:** Start with a foundational course focusing on **{foundational_course['skills'].split()[0]}** to bridge this gap.")
                
                st.markdown("#### Foundational Course Recommendation:")
                display_course_card(foundational_course, show_similarity=False, show_completion_button=True)
        else:
            st.success("🎉 **Great Job!** Your current skills align well with your chosen interest domains. You are ready for advanced topics!")
    else:
        st.info("ℹ️ Select an **Interest Domain** in your User Profile to enable Skill Gap Analysis.")
    # --- END AI SKILL GAP ANALYSIS ---
    
    st.markdown("---")
    if st.session_state.last_recommendations:
        st.subheader("🎯 Top Recommended Courses")
        
        for i, course in enumerate(st.session_state.last_recommendations, 1):
            display_course_card(course, show_similarity=True, show_completion_button=True)
            
    elif 'last_recommendations' in st.session_state:
        st.info("🤔 No courses found matching your criteria. Try widening your difficulty filter.")


def show_browse_page():
    st.title("📚 Browse All Courses")
    st.markdown("---")
    
    st.subheader("🔍 Filter & Sort Courses")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.selectbox(
            "**Domain/Category**",
            ["All"] + sorted(st.session_state.recommender.df['category'].unique().tolist())
        )
    
    with col2:
        platform_filter = st.selectbox(
            "**Platform**",
            ["All"] + sorted(st.session_state.recommender.df['platform'].unique().tolist())
        )
    
    with col3:
        sort_by = st.selectbox(
            "**Sort By**",
            options=['Rating (High to Low)', 'Duration (Shortest First)'],
            index=0 
        )
        
    rating_filter = st.slider(
        "**Minimum Rating (⭐)**",
        min_value=3.0, max_value=5.0, value=0.0, step=0.1
    )
    
    filtered_df = st.session_state.recommender.df.copy()
    
    if category_filter != "All": filtered_df = filtered_df[filtered_df['category'] == category_filter]
    if platform_filter != "All": filtered_df = filtered_df[filtered_df['platform'] == platform_filter]
    if rating_filter > 0.0: filtered_df = filtered_df[filtered_df['rating'] >= rating_filter]
    
    if sort_by == 'Rating (High to Low)':
        filtered_df = filtered_df.sort_values(by='rating', ascending=False)
    elif sort_by == 'Duration (Shortest First)':
        filtered_df = filtered_df.sort_values(by='duration', ascending=True)

    st.markdown("---")
    st.subheader(f"📖 Found {len(filtered_df)} Courses")
    
    if len(filtered_df) == 0:
        st.info("🤷 No courses found matching your filters. Try adjusting your criteria.")
        return
    
    for _, course in filtered_df.iterrows():
        display_course_card(course.to_dict(), show_completion_button=True)


def show_bookmarks_page():
    st.title("⭐ My Bookmarked Courses")
    st.markdown("---")
    
    bookmarked_courses = []
    for course_id in st.session_state.user_profile['bookmarks']:
        course = st.session_state.recommender.df[
            st.session_state.recommender.df['course_id'] == course_id
        ]
        if not course.empty:
            bookmarked_courses.append(course.iloc[0].to_dict())
    
    if bookmarked_courses:
        st.success(f"🎉 You have **{len(bookmarked_courses)}** bookmarked courses!")
        
        for course in bookmarked_courses:
            display_course_card(course, show_completion_button=True)
            
        st.markdown("---")
        if st.button("🗑️ Clear All Bookmarks", type="secondary"):
            st.session_state.user_profile['bookmarks'] = []
            st.toast("🧹 All bookmarks cleared!", icon="🗑️")
            st.rerun() 
            
    else:
        st.info("""
        📚 **You haven't bookmarked any courses yet!**
        
        Go to **Get Recommendations** or **Browse Courses** and click the **⭐ Bookmark** button to save your favorites.
        """)

# --- Main App Execution ---

def main():
    st.set_page_config(
        page_title="CourseRec AI: Recommendation System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    def nav_to(page_name):
        st.session_state.page = page_name

    # Sidebar (Attractive Navbar Implementation)
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🎓 CourseRec AI</h2>", unsafe_allow_html=True)
        st.markdown("### Intelligent Learning Recommender")
        st.markdown("---")
        
        # Navigation Buttons
        st.markdown("#### Navigation")
        if st.button("🏠 Home", use_container_width=True, key="nav_home"): nav_to("Home")
        if st.button("👤 User Profile", use_container_width=True, key="nav_profile"): nav_to("User Profile")
        if st.button("💡 Get Recommendations", use_container_width=True, key="nav_recs"): nav_to("Recommendations")
        if st.button("📚 Browse Courses", use_container_width=True, key="nav_browse"): nav_to("Browse")
        if st.button("⭐ Bookmarks", use_container_width=True, key="nav_bookmarks"): nav_to("Bookmarks")
        
        st.markdown("---")
        
        # Quick user info
        profile = st.session_state.user_profile
        if profile.get('name'):
            st.markdown(f"**👋 Hello, {profile['name']}!**")
            st.markdown(f"**Focus:** `{', '.join(profile['interest_domains']) or 'N/A'}`")
            st.markdown(f"**⭐ Bookmarks:** {len(profile['bookmarks'])} | **✅ Completed:** {len(profile['completed_courses'])}")
            
        st.markdown("---")
        st.markdown("### ℹ️ System Status")
        st.info(f"""
        **Courses:** {len(st.session_state.recommender.df)}
        **Skills Tracked:** {len(ALL_TECHNICAL_SKILLS)}
        **Engine:** TF-IDF & Cosine Similarity
        """)
    
    # Page routing based on session state
    if st.session_state.page == "Home":
        show_home_page()
    elif st.session_state.page == "User Profile":
        show_profile_page()
    elif st.session_state.page == "Recommendations":
        show_recommendations_page()
    elif st.session_state.page == "Browse":
        show_browse_page()
    elif st.session_state.page == "Bookmarks":
        show_bookmarks_page()

if __name__ == "__main__":
    main()
