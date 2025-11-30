"""
AI Data Analyzer - Student Performance Analysis
Author: Sandra Mkanyi | Registration: 250618DAI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="AI Data Analyzer", page_icon="📊", layout="wide")

# Title
st.title("📊 AI Data Analyzer - Student Performance Analysis")
st.markdown("**Author: Sandra Mkanyi | Registration: 250618DAI**")
st.divider()

# Sidebar
st.sidebar.header("📁 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded! {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Show raw data
        with st.expander("👀 View Raw Data"):
            st.dataframe(data)
        
        st.divider()
        
        # Data Cleaning
        st.subheader("🧹 Data Cleaning")
        initial_rows = len(data)
        data = data.drop_duplicates()
        duplicates_removed = initial_rows - len(data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duplicate Rows Removed", duplicates_removed)
        with col2:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Fill missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mean(), inplace=True)
        
        st.divider()
        
        # Statistical Analysis
        st.subheader("📈 Statistical Analysis")
        
        # Calculate average score if subject columns exist
        if all(col in data.columns for col in ['Math_Score', 'Science_Score', 'English_Score']):
            data['Average_Score'] = data[['Math_Score', 'Science_Score', 'English_Score']].mean(axis=1)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Average", f"{data['Average_Score'].mean():.2f}")
            with col2:
                st.metric("Highest Score", f"{data['Average_Score'].max():.2f}")
            with col3:
                st.metric("Lowest Score", f"{data['Average_Score'].min():.2f}")
            with col4:
                st.metric("Std Deviation", f"{data['Average_Score'].std():.2f}")
            
            # Performance categories
            data['Performance'] = pd.cut(
                data['Average_Score'],
                bins=[0, 70, 80, 90, 100],
                labels=['Needs Improvement', 'Good', 'Very Good', 'Excellent']
            )
        
        # Show descriptive statistics
        st.markdown("### Descriptive Statistics")
        st.dataframe(data.describe().round(2))
        
        st.divider()
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Distribution", "Subject Comparison", "Relationships"])
        
        with tab1:
            if 'Average_Score' in data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    ax1.hist(data['Average_Score'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
                    ax1.axvline(data['Average_Score'].mean(), color='red', linestyle='--', 
                               label=f'Mean: {data["Average_Score"].mean():.1f}')
                    ax1.set_xlabel('Average Score')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('Score Distribution')
                    ax1.legend()
                    st.pyplot(fig1)
                    plt.close()
                
                with col2:
                    if 'Performance' in data.columns:
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        perf_counts = data['Performance'].value_counts()
                        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                        ax2.pie(perf_counts, labels=perf_counts.index, autopct='%1.1f%%',
                               colors=colors, startangle=90)
                        ax2.set_title('Performance Categories')
                        st.pyplot(fig2)
                        plt.close()
        
        with tab2:
            if all(col in data.columns for col in ['Math_Score', 'Science_Score', 'English_Score']):
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                subjects = ['Math', 'Science', 'English']
                avg_scores = [
                    data['Math_Score'].mean(),
                    data['Science_Score'].mean(),
                    data['English_Score'].mean()
                ]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                bars = ax3.bar(subjects, avg_scores, color=colors, alpha=0.8, edgecolor='black')
                ax3.set_ylabel('Average Score')
                ax3.set_title('Average Performance by Subject')
                ax3.set_ylim(0, 100)
                
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom')
                st.pyplot(fig3)
                plt.close()
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Study_Hours' in data.columns and 'Average_Score' in data.columns:
                    fig4, ax4 = plt.subplots(figsize=(8, 5))
                    ax4.scatter(data['Study_Hours'], data['Average_Score'],
                               c=data['Average_Score'], cmap='viridis', s=100, alpha=0.6)
                    ax4.set_xlabel('Study Hours per Day')
                    ax4.set_ylabel('Average Score')
                    ax4.set_title('Study Hours vs Performance')
                    
                    z = np.polyfit(data['Study_Hours'], data['Average_Score'], 1)
                    p = np.poly1d(z)
                    ax4.plot(data['Study_Hours'], p(data['Study_Hours']), "r--", linewidth=2)
                    st.pyplot(fig4)
                    plt.close()
            
            with col2:
                if 'Attendance' in data.columns and 'Average_Score' in data.columns:
                    fig5, ax5 = plt.subplots(figsize=(8, 5))
                    ax5.scatter(data['Attendance'], data['Average_Score'],
                               c='coral', s=100, alpha=0.6, edgecolors='black')
                    ax5.set_xlabel('Attendance (%)')
                    ax5.set_ylabel('Average Score')
                    ax5.set_title('Attendance vs Performance')
                    
                    z = np.polyfit(data['Attendance'], data['Average_Score'], 1)
                    p = np.poly1d(z)
                    ax5.plot(data['Attendance'], p(data['Attendance']), "b--", linewidth=2)
                    st.pyplot(fig5)
                    plt.close()
        
        st.divider()
        
        # Insights
        st.subheader("🤖 AI-Generated Insights")
        
        if 'Average_Score' in data.columns:
            avg_score = data['Average_Score'].mean()
            st.markdown(f"**Overall Performance:** {avg_score:.2f}/100")
            
            if avg_score >= 85:
                st.success("→ Students are performing excellently! 🎉")
            elif avg_score >= 75:
                st.info("→ Good performance with room for improvement 📈")
            else:
                st.warning("→ Performance needs attention 📚")
        
        if 'Study_Hours' in data.columns and 'Average_Score' in data.columns:
            correlation = data['Study_Hours'].corr(data['Average_Score'])
            st.markdown(f"**Study Hours Impact:** Correlation = {correlation:.3f}")
            
            if correlation > 0.6:
                st.success("→ Strong positive correlation detected! ⏰")
            else:
                st.info("→ Moderate impact from study hours 🤔")
        
        if all(col in data.columns for col in ['Math_Score', 'Science_Score', 'English_Score']):
            subject_avgs = {
                'Math': data['Math_Score'].mean(),
                'Science': data['Science_Score'].mean(),
                'English': data['English_Score'].mean()
            }
            strongest = max(subject_avgs, key=subject_avgs.get)
            weakest = min(subject_avgs, key=subject_avgs.get)
            
            st.markdown(f"**Strongest Subject:** {strongest} ({subject_avgs[strongest]:.1f}) 💪")
            st.markdown(f"**Weakest Subject:** {weakest} ({subject_avgs[weakest]:.1f}) - Focus here! 📝")
        
        st.divider()
        
        # Download
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Analyzed Data",
            data=csv,
            file_name='analyzed_data.csv',
            mime='text/csv'
        )
        
        st.success("✅ Analysis Complete!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure your CSV has the correct column names!")

else:
    st.info("👆 Upload a CSV file to start analysis")
    st.markdown("""
    ### Expected CSV Format:
    - `Name` - Student names
    - `Math_Score` - Math scores (0-100)
    - `Science_Score` - Science scores (0-100)  
    - `English_Score` - English scores (0-100)
    - `Study_Hours` - Study hours per day
    - `Attendance` - Attendance percentage
    """)