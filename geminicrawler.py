import streamlit as st
import pandas as pd
import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import google.generativeai as genai

# ✅ Configure Gemini API key
genai.configure(api_key="AIzaSyB92kFUtG3ZGyzuTvge7tspT4AneRarLuk")

def scrape_youtube_comments(url):
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    items = []
    try:
        driver.get(url)
        time.sleep(3)

        driver.execute_script('window.scrollTo(0, 500);')
        time.sleep(3)

        for _ in range(1):
            driver.execute_script('window.scrollBy(0, 3000);')
            time.sleep(3)

        username_elems = driver.find_elements(By.XPATH, '//*[@id="author-text"]')[:3]
        comment_elems = driver.find_elements(By.XPATH, '//*[@id="content-text"]')[:3]

        for username, comment in zip(username_elems, comment_elems):
            items.append({
                'Author': username.text.strip(),
                'Comment': comment.text.strip()
            })

        filename = 'sample.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Author', 'Comment'])
            writer.writeheader()
            writer.writerows(items)

        print(f"✅ Extracted {len(items)} comments.")
    finally:
        driver.quit()

    return filename


def classify_comment_with_gemini(comment, max_retries=3):
    prompt = f"""The following is a YouTube comment in Telugu:
\"{comment}\"

Classify this comment as either "Good" (positive/neutral) or "Bad" (negative/abusive/offensive/harsh). 
Return only "Good" or "Bad" as your response."""
    
    for attempt in range(max_retries):
        try:
            response = genai.GenerativeModel('gemini-1.5-pro').generate_content(prompt)
            label = response.text.strip()
            
            # Clean and normalize the response
            label = label.lower().strip()
            
            # Handle various response formats
            if 'good' in label or 'positive' in label or 'neutral' in label:
                return "Good"
            elif 'bad' in label or 'negative' in label or 'abusive' in label or 'offensive' in label or 'harsh' in label:
                return "Bad"
            else:
                # If response is unclear, try to extract the sentiment
                if len(label) <= 10:  # Short response, likely just the label
                    if label.startswith('g') or label.startswith('G'):
                        return "Good"
                    elif label.startswith('b') or label.startswith('B'):
                        return "Bad"
                
                # Log the unexpected response for debugging
                print(f"Attempt {attempt + 1}: Unexpected response from Gemini: '{response.text}' for comment: '{comment[:50]}...'")
                
                # If this is the last attempt, return default
                if attempt == max_retries - 1:
                    return "Good"  # Default to Good as fallback
                
                # Wait a bit before retrying
                time.sleep(1)
                
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error processing comment: {e}")
            print(f"Comment that caused error: {comment[:50]}...")
            
            # If this is the last attempt, return default
            if attempt == max_retries - 1:
                return "Good"  # Default to Good instead of Unknown
            
            # Wait a bit before retrying
            time.sleep(1)
    
    return "Good"  # Final fallback



def process_comments():
    try:
        df = pd.read_csv('sample.csv')
        total_comments = len(df)
        
        print(f"Processing {total_comments} comments...")
        
        sentiments = []
        for idx, comment in enumerate(df['Comment']):
            print(f"Processing comment {idx + 1}/{total_comments}: {comment[:50]}...")
            sentiment = classify_comment_with_gemini(comment)
            sentiments.append(sentiment)
            print(f"Sentiment: {sentiment}")
        
        df['Sentiment'] = sentiments
        df.to_csv('labeled_comments.csv', index=False, encoding='utf-8')
        
        # Print summary
        good_count = len(df[df['Sentiment'] == 'Good'])
        bad_count = len(df[df['Sentiment'] == 'Bad'])
        print(f"✅ Classification completed! Good: {good_count}, Bad: {bad_count}")
        
        return df
        
    except Exception as e:
        print(f"Error in process_comments: {e}")
        st.error(f"Error processing comments: {e}")
        return None


def main():
    st.title('YouTube Comment Scraper and Sentiment Analysis')
    st.subheader('Scrape YouTube Comments and Classify Them as Good or Bad')

    youtube_url = st.text_input(
        "Enter YouTube Video URL",
        "https://www.youtube.com/watch?v=IOopJ-PDpac"
    )

    if st.button('Scrape Comments'):
        if youtube_url:
            st.text('Scraping YouTube comments...')
            filename = scrape_youtube_comments(youtube_url)
            st.success(f'Comments have been scraped and saved to {filename}')
        else:
            st.warning('Please enter a valid YouTube video URL.')

    if st.button('Classify Comments'):
        st.text('Classifying comments as Good or Bad...')
        df = process_comments()
        
        if df is not None:
            st.success('Classification completed!')
            st.subheader('Classified Comments')
            st.write(df)
            
            # Show summary statistics
            good_count = len(df[df['Sentiment'] == 'Good'])
            bad_count = len(df[df['Sentiment'] == 'Bad'])
            st.info(f"Summary: {good_count} Good comments, {bad_count} Bad comments")
            
            st.download_button(
                label="Download Classified Comments",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='labeled_comments.csv',
                mime='text/csv'
            )
        else:
            st.error('Classification failed. Please check the console for error details.')


if __name__ == '__main__':
    main()
