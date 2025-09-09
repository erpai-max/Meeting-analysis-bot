# Meeting-Analysis-Reports
A reporting and analytics system for meeting performance tracking. Automates data collection, generates actionable insights, and delivers structured dashboards to improve team productivity and decision-making.
The script follows a clear, step-by-step process for each team member's folder. Think of it like an assembly line:

Search & Find 🕵️‍♀️: The script logs into Google Drive and goes to the first team member's folder (e.g., Sharath's). It then searches for any new audio or video files that haven't been processed yet.

Download & Transcribe 🎤: If it finds a new file, it downloads it to the temporary memory of the GitHub runner. It then uses the faster-whisper model to listen to the entire recording and convert all the spoken words into a plain text transcript.

Analyze with AI 🧠: This is the core step. The script takes the text transcript and combines it with the detailed ERP and ASP product information you provided. It sends this entire package to the Gemini AI with the instruction: "Read this conversation, and using the product context I gave you, fill out these 47 data points in a JSON format."

Write to Sheet ✍️: The script receives the structured JSON data back from Gemini. It then connects to your Google Sheet, reads the headers in row 1, and appends a new row with the 47 points of analysis in the correct columns.

Move & Clean Up 📂: To ensure the same file is never analyzed again, the script moves the original audio/video file from the team member's folder to your "Processed" folder. This marks the job as complete.

The script repeats this entire process for every folder in your TEAM_FOLDERS list.
