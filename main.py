import os
import io
import json
import tempfile
import logging
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.generativeai as genai
import gspread

# --- Dependencies ---
from faster_whisper import WhisperModel

# --- Configuration ---
GCP_SERVICE_ACCOUNT_KEY = os.environ.get("GCP_SA_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# !!! IMPORTANT: VERIFY THESE IDs ARE CORRECT !!!
GOOGLE_SHEET_ID = "12vuVG0jTs5GUFaywj3piz6zk44X92JeB"
PROCESSED_FOLDER_ID = "1fI7QAr1MwJlh4FTJGezb5LTXGDqMpmXT"

# --- Helper Function ---
def get_id_from_url(url):
    return url.split('/')[-1].split('?')[0]

TEAM_FOLDERS = {
    "Sharath": get_id_from_url("https://drive.google.com/drive/folders/1Ddvswz50pgniX3WTQdV-gd2wEz56di5H?usp=sharing"),
    "Tavish": get_id_from_url("https://drive.google.com/drive/folders/1ug6CVE96qLJNfmwLdC631Zz5KSyVO9FK?usp=sharing"),
    "Sripal": get_id_from_url("https://drive.google.com/drive/folders/19F77ZKyCQgEb43HidnmFMrPxT9Z6NqcW?usp=sharing"),
    "Musthafa": get_id_from_url("https://drive.google.com/drive/folders/1kwg9bnZq_CgKE0YKXfDqnMvJHVz7VI3e?usp=sharing"),
    "Hemanth": get_id_from_url("https://drive.google.com/drive/folders/148aNehlAkHFL_gpX73Ko61ftg0XgE_ye?usp=sharing"),
    "Darshan": get_id_from_url("https://drive.google.com/drive/folders/1BciBnzjxbeCnPlPsTCVVhv1NH2BxmIGl?usp=sharing"),
    "Vishal": get_id_from_url("https://drive.google.com/drive/folders/1ZFE5sAVMOlJcHC5r3LEVZzbEnZzOvY-G?usp=sharing"),
    "Rahul": get_id_from_url("https://drive.google.com/drive/folders/1kxThyuSLuMUmM3GFF4H7PcVx8yUzasix?usp=sharing"),
    "Akshay": get_id_from_url("https://drive.google.com/drive/folders/13gXYRgpi4xzhLMWppa8ZI2Ljr_wZHAQF?usp=sharing")
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FUNCTION DEFINITIONS ---

def authenticate_google_services():
    logging.info("Attempting to authenticate with Google services...")
    try:
        if not GCP_SERVICE_ACCOUNT_KEY:
            logging.error("CRITICAL: GCP_SA_KEY environment variable not found.")
            return None, None
            
        creds_info = json.loads(GCP_SERVICE_ACCOUNT_KEY)
        scopes = [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
        ]
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
        
        drive_service = build("drive", "v3", credentials=creds)
        
        # --- THIS IS THE CORRECTED LINE ---
        gc = gspread.authorize(creds)
        
        logging.info("SUCCESS: Authentication with Google services complete.")
        return drive_service, gc
    except Exception as e:
        logging.error(f"CRITICAL: Authentication failed: {e}")
        return None, None

def download_file(drive_service, file_id):
    logging.info(f"Starting download for file ID: {file_id}")
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            logging.info(f"Download progress: {int(status.progress() * 100)}%")
    fh.seek(0)
    logging.info("SUCCESS: File download complete.")
    return fh

def transcribe_audio(file_content, original_filename):
    logging.info("Starting transcription process...")
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(original_filename)[1], delete=False) as temp_file:
        temp_file.write(file_content.read())
        temp_file_path = temp_file.name

    try:
        logging.info("Initializing Whisper model (tiny.en)...")
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        
        logging.info(f"Transcribing {temp_file_path}...")
        segments, _ = model.transcribe(temp_file_path, beam_size=5)
        
        transcript = " ".join(segment.text for segment in segments)
        
        logging.info(f"SUCCESS: Transcription completed. Transcript length: {len(transcript)} characters.")
        return transcript.strip()
    except Exception as e:
        logging.error(f"ERROR: Transcription failed: {e}")
        return None
    finally:
        os.remove(temp_file_path)
        logging.info(f"Cleaned up temporary file: {temp_file_path}")

def analyze_transcript_with_gemini(transcript, owner_name):
    logging.info("Starting analysis with Gemini...")
    if not GEMINI_API_KEY:
        logging.error("CRITICAL: GEMINI_API_KEY environment variable not found.")
        return None

    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        ### ROLE AND GOAL ###
        You are an expert sales meeting analyst for our company, specializing in Society Management software. Your goal is to meticulously analyze a sales meeting transcript, score the performance of the sales representative '{owner_name}', and extract key business information. Your analysis must differentiate whether the primary product discussed was our **ERP** solution or our **ASP** (Accounting Services as a Product) offering.

        ### CONTEXT: PRODUCT AND PRICING INFORMATION ###
        ---
        **Product 1: ERP (Enterprise Resource Planning)**
        This is our comprehensive, self-service software solution for society management.
        * **Pricing:** ₹12 + 18% GST per flat, per month.
        * **Key Differentiators & Features:**
            * **Financial & Accounting:** Instant settlement of payments, minimal gateway charges, full Tally integration (import/export), in-house payment gateway, superfast data migration, generation of all key financial reports (Balance Sheet, TDS, GST, etc.), e-invoicing, bank reconciliation (MT940), vendor accounting dashboard, budgeting, and bill ageing reports.
            * **Billing Automation:** 350+ bill combinations, bill scheduler, 2-level maker-checker approval system, automated reminders, customer interest calculations, proforma invoicing, meter reading uploads for automated invoices, and late fee calculation.
            * **Management & Operations:** Comprehensive property management, inventory management, asset management with QR code tagging, purchase order approval process, preventive planned maintenance (PPM) reminders, and date & time stamps for vendor entry.
            * **Resident Features:** Virtual accounts for payments.
            * **Security & Access:** Role-based approval and access controls.

        **Product 2: ASP (Accounting Services as a Product)**
        This is a managed service where we handle the society's accounting for them using our software. It's a "done-for-you" service.
        * **Pricing:** ₹22.5 + 18% GST per flat, per month.
        * **Scope of Work & Offerings:**
            * A dedicated accountant is provided for virtual support.
            * We handle all computerized online billing and receipt generation.
            * We manage book-keeping for all incomes & expenses.
            * We perform bank reconciliation and follow up on suspense entries.
            * We provide system-generated, non-audited financial reports.
            * We assist with the finalization of accounts and coordinate with auditors.
            * Includes all community and visitor management system features.
            * Also includes vendor, purchase order, inventory, and amenities booking management.
            * Software access and data backups are included.
            * **Crucial Note:** This is a paid accounting service. The scope of work changes if the society decides to purchase the ERP software instead.

        ---
        ### INPUT: MEETING TRANSCRIPT ###
        ---
        {transcript}
        ---

        ### TASK AND INSTRUCTIONS ###
        Analyze the provided meeting transcript based on the product context above. First, determine if the meeting was primarily about **ERP**, **ASP**, or a combination of both. Then, extract the following 47 data points. For scoring, evaluate how well the sales rep pitched the relevant product's features and handled objections. If a specific piece of information is not mentioned, you MUST return "N/A".

        Your output MUST be a single, valid JSON object.

        ### REQUIRED JSON OUTPUT FORMAT ###
        {{
            "Date": "...", "POC Name": "...", "Society Name": "...", "Visit Type": "...",
            "Meeting Type": "e.g., ERP Pitch, ASP Pitch, ERP & ASP, General Inquiry",
            "Amount Value": "Extract any discussed price per flat, total value, or package cost.",
            "Months": "...",
            "Deal Status": "e.g., Hot Lead, Warm Lead, Cold Lead, Negotiation, Demo Scheduled",
            "Vendor Leads": "...", "Society Leads": "...",
            "Opening Pitch Score": "1-10 on rapport and agenda setting.",
            "Product Pitch Score": "1-10, based on how well the key features of the discussed product (ERP or ASP) were explained.",
            "Cross-Sell / Opportunity Handling": "1-10, did the rep correctly identify the client's need for ERP vs. ASP? Was an upsell/cross-sell attempted?",
            "Closing Effectiveness": "1-10 on defining clear next steps.",
            "Negotiation Strength": "1-10, how well were objections about price or features handled using our differentiators?",
            "Overall Sentiment": "Positive, Neutral, Negative, Mixed",
            "Total Score": "Sum of the 5 scores above (out of 50).",
            "% Score": "Total Score / 50, as a percentage.",
            "Risks / Unresolved Issues": "...", "Improvements Needed": "...", "Owner": "{owner_name}",
            "Email Id": "...", "Kibana ID": "...", "Manager": "...",
            "Product Pitch": "Summarize the pitch delivered. Was it clearly for ERP or ASP?",
            "Team": "...", "Media Link": "...", "Doc Link": "...",
            "Suggestions & Missed Topics": "What key features (e.g., 'Instant Settlement' for ERP, 'Dedicated Accountant' for ASP) were missed?",
            "Pre-meeting brief": "...", "Meeting duration (min)": "...",
            "Rebuttal Handling": "Describe how objections were handled.",
            "Rapport Building": "...", "Improvement Areas": "...",
            "Product Knowledge Displayed": "How well did the rep explain the features relevant to the client's needs?",
            "Call Effectiveness and Control": "...", "Next Step Clarity and Commitment": "...",
            "Missed Opportunities": "e.g., Client seemed overwhelmed, should have pitched ASP instead of ERP.",
            "Key Discussion Points": "...", "Key Questions": "...", "Competition Discussion": "...", "Action items": "...",
            "Positive Factors": "...", "Negative Factors": "...",
            "Customer Needs": "What were the client's main pain points? (e.g., Tally issues, reconciliation problems, billing complexity).",
            "Overall Client Sentiment": "...",
            "Feature Checklist Coverage": "List the key features mentioned from the relevant product (ERP or ASP)."
        }}
        """
        
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        logging.info("SUCCESS: Analysis with Gemini complete.")
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"ERROR: Gemini API analysis failed: {e}")
        return None

def write_to_google_sheets(gsheets_client, data):
    logging.info("Attempting to write data to Google Sheets...")
    try:
        spreadsheet = gsheets_client.open_by_key(GOOGLE_SHEET_ID)
        worksheet = spreadsheet.get_worksheet(0)
        
        headers = worksheet.row_values(1)
        if not headers:
            logging.warning("No headers found in the Google Sheet. Writing data keys as headers first.")
            header_keys = ["Date", "POC Name", "Society Name", "Visit Type", "Meeting Type", "Amount Value", "Months", "Deal Status", "Vendor Leads", "Society Leads", "Opening Pitch Score", "Product Pitch Score", "Cross-Sell / Opportunity Handling", "Closing Effectiveness", "Negotiation Strength", "Overall Sentiment", "Total Score", "% Score", "Risks / Unresolved Issues", "Improvements Needed", "Owner", "Email Id", "Kibana ID", "Manager", "Product Pitch", "Team", "Media Link", "Doc Link", "Suggestions & Missed Topics", "Pre-meeting brief", "Meeting duration (min)", "Rebuttal Handling", "Rapport Building", "Improvement Areas", "Product Knowledge Displayed", "Call Effectiveness and Control", "Next Step Clarity and Commitment", "Missed Opportunities", "Key Discussion Points", "Key Questions", "Competition Discussion", "Action items", "Positive Factors", "Negative Factors", "Customer Needs", "Overall Client Sentiment", "Feature Checklist Coverage"]
            worksheet.append_row(header_keys, value_input_option="USER_ENTERED")
            headers = header_keys

        row_to_insert = [data.get(header, "N/A") for header in headers]
        
        worksheet.append_row(row_to_insert, value_input_option="USER_ENTERED")
        logging.info(f"SUCCESS: Data for '{data.get('Society Name', 'N/A')}' written to Google Sheets.")
    except Exception as e:
        logging.error(f"ERROR: Failed to write to Google Sheets: {e}")

def move_file_to_processed(drive_service, file_id, source_folder_id):
    logging.info(f"Attempting to move file {file_id} to processed folder...")
    try:
        drive_service.files().update(
            fileId=file_id,
            addParents=PROCESSED_FOLDER_ID,
            removeParents=source_folder_id,
            fields="id, parents"
        ).execute()
        logging.info(f"SUCCESS: File {file_id} moved to processed folder.")
    except Exception as e:
        logging.error(f"ERROR: Failed to move file {file_id}: {e}")

def main():
    logging.info("--- Starting main execution ---")
    if not GOOGLE_SHEET_ID or not PROCESSED_FOLDER_ID:
        logging.error("CRITICAL: GOOGLE_SHEET_ID or PROCESSED_FOLDER_ID not set in the script.")
        return

    drive_service, gsheets_client = authenticate_google_services()
    if not drive_service or not gsheets_client:
        logging.error("CRITICAL: Exiting due to authentication failure.")
        return

    logging.info("Starting to check team folders...")
    for member_name, folder_id in TEAM_FOLDERS.items():
        logging.info(f"--- Checking folder for team member: {member_name} (Folder ID: {folder_id}) ---")
        try:
            query = f"'{folder_id}' in parents and (mimeType contains 'audio/' or mimeType contains 'video/')"
            logging.info(f"Executing Drive search query: {query}")
            results = drive_service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get("files", [])
            
            logging.info(f"Found {len(files)} new file(s) for {member_name}.")
            
            if not files:
                continue
            
            for file in files:
                file_id, file_name = file["id"], file["name"]
                logging.info(f"--- Processing file: {file_name} (ID: {file_id}) ---")
                try:
                    file_content = download_file(drive_service, file_id)
                    transcript = transcribe_audio(file_content, file_name)
                    
                    if transcript:
                        analysis_data = analyze_transcript_with_gemini(transcript, member_name)
                        if analysis_data:
                            write_to_google_sheets(gsheets_client, analysis_data)
                            move_file_to_processed(drive_service, file_id, folder_id)
                        
                        time.sleep(20) # Pauses for 20 seconds to help with API rate limits
                except Exception as e:
                    logging.error(f"ERROR processing file {file_name}: {e}")
        
        except Exception as e:
            logging.error(f"CRITICAL ERROR while processing {member_name}'s folder: {e}")
    
    logging.info("--- Main execution finished ---")

if __name__ == "__main__":
    main()
