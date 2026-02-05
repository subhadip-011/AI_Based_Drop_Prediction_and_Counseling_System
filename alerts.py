import smtplib 
import ssl
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

def send_email_alert(student, recipients, risk="High", custom_message=None,
                     sender_email="pansubhadip779@gmail.com", sender_password="fxymavqtpgtxbowa"):
    """
    Send an email alert about a student at risk, or send a custom feedback message.
    """

    try:
        if not sender_email or not sender_password:
            raise ValueError("Sender email and password must be provided.")

        # Convert recipients to list
        if isinstance(recipients, str):
            recipients = [recipients]

        # Subject
        if risk == "Feedback":
            subject = "Student Feedback Form Request"
        else:
            subject = f"Student Dropout Risk Alert ({risk})"

        # Body
        if custom_message:
            body = custom_message
        else:
            body = f"""
            Dear Guardian/Mentor,

            This is an automated alert from the Student Counseling System.

            Student ID: {student.get('student_id', 'N/A')}
            Attendance: {student.get('attendance_pct', 'N/A')}%
            Avg Score (last 3 tests): {student.get('avg_score_last_3', 'N/A')}
            Overdue Days: {student.get('overdue_days', 'N/A')}
            Risk Level: {student.get('Risk_Level', 'N/A')}

            Action Required: Please schedule a counseling session with the student as early as possible.

            Regards,
            Counseling Dashboard
            """

        # Build email
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Secure connection + send
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())

        # Log entry
        log_entry = {
            "student_id": student.get("student_id", "N/A"),
            "risk": risk,
            "recipients": "; ".join(recipients),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Sent"
        }

        # Save log
        log_file = "feedback_log.csv" if risk == "Feedback" else "alerts_log.csv"
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([log_entry])
        df.to_csv(log_file, index=False)

        return log_entry

    except Exception as e:
        return {
            "student_id": student.get("student_id", "N/A"),
            "risk": risk,
            "recipients": recipients,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": f"Failed: {e}"
        }
