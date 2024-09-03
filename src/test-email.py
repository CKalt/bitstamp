import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json


# Load configuration from JSON file
try:
    with open('.ses_smtp_config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("Error: .ses_smtp_config.json file not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON in .ses_smtp_config.json file.")
    exit(1)

# Extract configuration
SENDER = config.get('SENDER')
RECIPIENT = config.get('RECIPIENT')
SMTP_USERNAME = config.get('SMTP_USERNAME')
SMTP_PASSWORD = config.get('SMTP_PASSWORD')
HOST = config.get('HOST')
PORT = config.get('PORT')

# The subject line and body for the email
SUBJECT = "Test Email from EC2 via Amazon SES"
BODY_TEXT = "This is a test email sent from an EC2 instance using Amazon SES."
BODY_HTML = """<html>
<head></head>
<body>
  <h1>Test Email from EC2</h1>
  <p>This is a test email sent from an EC2 instance using Amazon SES.</p>
</body>
</html>
"""

# Create message container
message = MIMEMultipart('alternative')
message['Subject'] = SUBJECT
message['From'] = SENDER
message['To'] = RECIPIENT

# Record the MIME types of both parts - text/plain and text/html
part1 = MIMEText(BODY_TEXT, 'plain')
part2 = MIMEText(BODY_HTML, 'html')

# Attach parts into message container
message.attach(part1)
message.attach(part2)

# Try to send the email
try:
    with smtplib.SMTP(HOST, PORT) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SENDER, RECIPIENT, message.as_string())
    print("Email sent successfully")
except Exception as e:
    print(f"Error: {e}")
