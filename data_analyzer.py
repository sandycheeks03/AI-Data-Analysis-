"""
AI Data Analyzer - Student Performance Analysis
DAI011: Programming for AI - CAT 2
Author: Sandra Mkanyi
Registration: 250618DAI

This program analyzes student performance data using pandas, numpy, matplotlib, and seaborn.
It performs data cleaning, statistical analysis, and creates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(filepath):
    """
    Load dataset from CSV file
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        print("✓ Data loaded successfully!")
        print(f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns\n")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(data):
    """
    Clean the dataset by handling missing values and duplicates
    
    Parameters:
    data (DataFrame): Raw dataset
    
    Returns:
    DataFrame: Cleaned dataset
    """
    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)
    
    # Remove duplicates
    initial_rows = len(data)
    data = data.drop_duplicates()
    removed_duplicates = initial_rows - len(data)
    print(f"\n✓ Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values (if any) - fill with mean for numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mean(), inplace=True)
            print(f"✓ Filled missing values in '{col}' with mean")
    
    print(f"\n✓ Data cleaning complete! Clean dataset has {len(data)} rows\n")
    return data


def perform_statistical_analysis(data):
    """
    Perform descriptive statistical analysis on the dataset
    
    Parameters:
    data (DataFrame): Cleaned dataset
    
    Returns:
    dict: Dictionary containing various statistics
    """
    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Select numeric columns only
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Basic statistics
    print("\n1. DESCRIPTIVE STATISTICS:")
    print("-" * 60)
    print(numeric_data.describe().round(2))
    
    # Calculate average scores
    print("\n2. AVERAGE PERFORMANCE METRICS:")
    print("-" * 60)
    if all(col in data.columns for col in ['Math_Score', 'Science_Score', 'English_Score']):
        data['Average_Score'] = data[['Math_Score', 'Science_Score', 'English_Score']].mean(axis=1)
        print(f"Overall Average Score: {data['Average_Score'].mean():.2f}")
        print(f"Highest Average Score: {data['Average_Score'].max():.2f}")
        print(f"Lowest Average Score: {data['Average_Score'].min():.2f}")
        print(f"Standard Deviation: {data['Average_Score'].std():.2f}")
    
    # Correlation analysis
    print("\n3. CORRELATION ANALYSIS:")
    print("-" * 60)
    if 'Study_Hours' in data.columns and 'Average_Score' in data.columns:
        correlation = data['Study_Hours'].corr(data['Average_Score'])
        print(f"Correlation between Study Hours and Average Score: {correlation:.3f}")
        
        if correlation > 0.7:
            print("→ Strong positive correlation detected!")
        elif correlation > 0.4:
            print("→ Moderate positive correlation detected")
        else:
            print("→ Weak correlation detected")
    
    # Performance categories
    print("\n4. PERFORMANCE CATEGORIZATION:")
    print("-" * 60)
    if 'Average_Score' in data.columns:
        data['Performance'] = pd.cut(data['Average_Score'], 
                                     bins=[0, 70, 80, 90, 100],
                                     labels=['Needs Improvement', 'Good', 'Very Good', 'Excellent'])
        print("\nPerformance Distribution:")
        print(data['Performance'].value_counts().sort_index())
    
    # Statistical insights dictionary
    stats = {
        'total_students': len(data),
        'average_score': data['Average_Score'].mean() if 'Average_Score' in data.columns else None,
        'top_performer': data.loc[data['Average_Score'].idxmax(), 'Name'] if 'Average_Score' in data.columns and 'Name' in data.columns else None,
        'correlation': correlation if 'Study_Hours' in data.columns and 'Average_Score' in data.columns else None
    }
    
    print("\n" + "=" * 60 + "\n")
    return stats, data


def create_visualizations(data):
    """
    Create multiple visualizations to understand the data better
    
    Parameters:
    data (DataFrame): Dataset with calculated metrics
    """
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Score Distribution (Histogram)
    plt.subplot(2, 3, 1)
    if 'Average_Score' in data.columns:
        plt.hist(data['Average_Score'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Average Score', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Distribution of Average Scores', fontsize=12, fontweight='bold')
        plt.axvline(data['Average_Score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {data["Average_Score"].mean():.1f}')
        plt.legend()
    
    # 2. Subject-wise Performance (Bar Chart)
    plt.subplot(2, 3, 2)
    if all(col in data.columns for col in ['Math_Score', 'Science_Score', 'English_Score']):
        subjects = ['Math', 'Science', 'English']
        avg_scores = [data['Math_Score'].mean(), data['Science_Score'].mean(), data['English_Score'].mean()]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = plt.bar(subjects, avg_scores, color=colors, alpha=0.8, edgecolor='black')
        plt.ylabel('Average Score', fontsize=10)
        plt.title('Average Performance by Subject', fontsize=12, fontweight='bold')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Study Hours vs Average Score (Scatter Plot)
    plt.subplot(2, 3, 3)
    if 'Study_Hours' in data.columns and 'Average_Score' in data.columns:
        plt.scatter(data['Study_Hours'], data['Average_Score'], 
                   c=data['Average_Score'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        plt.xlabel('Study Hours per Day', fontsize=10)
        plt.ylabel('Average Score', fontsize=10)
        plt.title('Study Hours vs Performance', fontsize=12, fontweight='bold')
        plt.colorbar(label='Score')
        
        # Add trend line
        z = np.polyfit(data['Study_Hours'], data['Average_Score'], 1)
        p = np.poly1d(z)
        plt.plot(data['Study_Hours'], p(data['Study_Hours']), "r--", alpha=0.8, linewidth=2)
    
    # 4. Performance Category Distribution (Pie Chart)
    plt.subplot(2, 3, 4)
    if 'Performance' in data.columns:
        performance_counts = data['Performance'].value_counts()
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        plt.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90, explode=[0.05] * len(performance_counts))
        plt.title('Performance Category Distribution', fontsize=12, fontweight='bold')
    
    # 5. Attendance vs Average Score (Scatter Plot)
    plt.subplot(2, 3, 5)
    if 'Attendance' in data.columns and 'Average_Score' in data.columns:
        plt.scatter(data['Attendance'], data['Average_Score'], 
                   c='coral', s=100, alpha=0.6, edgecolors='black')
        plt.xlabel('Attendance (%)', fontsize=10)
        plt.ylabel('Average Score', fontsize=10)
        plt.title('Attendance vs Performance', fontsize=12, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(data['Attendance'], data['Average_Score'], 1)
        p = np.poly1d(z)
        plt.plot(data['Attendance'], p(data['Attendance']), "b--", alpha=0.8, linewidth=2)
    
    # 6. Box Plot for Score Distribution
    plt.subplot(2, 3, 6)
    if all(col in data.columns for col in ['Math_Score', 'Science_Score', 'English_Score']):
        score_data = data[['Math_Score', 'Science_Score', 'English_Score']]
        box = plt.boxplot([score_data['Math_Score'], score_data['Science_Score'], 
                          score_data['English_Score']], 
                         labels=['Math', 'Science', 'English'],
                         patch_artist=True)
        
        # Color the boxes
        colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(box['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel('Score', fontsize=10)
        plt.title('Score Distribution by Subject (Box Plot)', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualizations created and saved as 'analysis_results.png'")
    plt.show()
    
    print("\n" + "=" * 60 + "\n")


def generate_insights(data, stats):
    """
    Generate AI-driven insights from the analysis
    
    Parameters:
    data (DataFrame): Analyzed dataset
    stats (dict): Statistical summary
    """
    print("=" * 60)
    print("AI-GENERATED INSIGHTS")
    print("=" * 60)
    
    print("\n📊 KEY FINDINGS:\n")
    
    # Insight 1: Overall Performance
    if stats['average_score']:
        print(f"1. The average student performance is {stats['average_score']:.2f}/100")
        if stats['average_score'] >= 85:
            print("   → Students are performing excellently overall!")
        elif stats['average_score'] >= 75:
            print("   → Students show good performance with room for improvement")
        else:
            print("   → Performance needs attention and improvement strategies")
    
    # Insight 2: Study Hours Impact
    if stats['correlation']:
        print(f"\n2. Study hours show a correlation of {stats['correlation']:.3f} with performance")
        if stats['correlation'] > 0.6:
            print("   → Strong evidence that more study hours lead to better scores!")
        else:
            print("   → Study hours have some impact, but other factors matter too")
    
    # Insight 3: Top Performer
    if stats['top_performer']:
        print(f"\n3. Top performing student: {stats['top_performer']}")
        print("   → This student can be a peer mentor for others")
    
    # Insight 4: Subject Analysis
    if all(col in data.columns for col in ['Math_Score', 'Science_Score', 'English_Score']):
        subject_avgs = {
            'Math': data['Math_Score'].mean(),
            'Science': data['Science_Score'].mean(),
            'English': data['English_Score'].mean()
        }
        strongest_subject = max(subject_avgs, key=subject_avgs.get)
        weakest_subject = min(subject_avgs, key=subject_avgs.get)
        
        print(f"\n4. Strongest subject: {strongest_subject} (avg: {subject_avgs[strongest_subject]:.1f})")
        print(f"   Weakest subject: {weakest_subject} (avg: {subject_avgs[weakest_subject]:.1f})")
        print(f"   → Focus improvement efforts on {weakest_subject}")
    
    # Insight 5: Attendance Importance
    if 'Attendance' in data.columns and 'Average_Score' in data.columns:
        att_corr = data['Attendance'].corr(data['Average_Score'])
        print(f"\n5. Attendance correlation with scores: {att_corr:.3f}")
        if att_corr > 0.5:
            print("   → Regular attendance significantly impacts performance!")
    
    print("\n" + "=" * 60 + "\n")


def main():
    """
    Main function to orchestrate the entire data analysis pipeline
    """
    print("\n" + "=" * 60)
    print("  AI DATA ANALYZER - STUDENT PERFORMANCE ANALYSIS")
    print("=" * 60 + "\n")
    
    # Step 1: Load Data
    filepath = 'sample_data.csv'
    data = load_data(filepath)
    
    if data is None:
        print("Exiting due to data loading error.")
        return
    
    # Step 2: Display first few rows
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\n")
    
    # Step 3: Clean Data
    clean_df = clean_data(data)
    
    # Step 4: Perform Statistical Analysis
    stats, analyzed_data = perform_statistical_analysis(clean_df)
    
    # Step 5: Create Visualizations
    create_visualizations(analyzed_data)
    
    # Step 6: Generate AI Insights
    generate_insights(analyzed_data, stats)
    
    # Step 7: Save processed data
    analyzed_data.to_csv('analyzed_data.csv', index=False)
    print("✓ Analyzed data saved to 'analyzed_data.csv'\n")
    
    print("=" * 60)
    print("  ANALYSIS COMPLETE!")
    print("=" * 60)


# Entry point
if __name__ == "__main__":
    main()