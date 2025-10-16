#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email Notification Module
Sends pipeline completion notifications with attachments
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Optional
import socket

from .config import Config
from .logger import PipelineLogger


class EmailSender:
    """Handles email notifications for pipeline completion"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
        
    def send_completion_email(
        self,
        success: bool,
        output_dir: Path,
        timing_report: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Send pipeline completion notification
        
        Args:
            success: Whether pipeline completed successfully
            output_dir: Path to output directory
            timing_report: Timing report text
            error_message: Error message if failed
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.config.email.enabled:
            self.logger.info("Email notifications disabled, skipping")
            return True
            
        try:
            with self.logger.timed_operation("Sending Email Notification"):
                # Create message
                msg = self._create_message(success, output_dir, timing_report, error_message)
                
                # Attach files
                self._attach_files(msg, output_dir, success)
                
                # Send email
                self._send_email(msg)
                
                self.logger.info(f"Email sent successfully to: {', '.join(self.config.email.to_addresses)}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}", exc_info=True)
            return False
    
    def _create_message(
        self,
        success: bool,
        output_dir: Path,
        timing_report: str,
        error_message: Optional[str]
    ) -> MIMEMultipart:
        """Create email message with HTML body"""
        
        msg = MIMEMultipart('alternative')
        msg['From'] = self.config.email.from_address
        msg['To'] = ', '.join(self.config.email.to_addresses)
        
        if self.config.email.cc_addresses:
            msg['Cc'] = ', '.join(self.config.email.cc_addresses)
        
        # Subject line
        status = "✅ SUCCESS" if success else "❌ FAILED"
        hostname = socket.gethostname()
        msg['Subject'] = f"{status} - Donations Analytics Pipeline [{hostname}]"
        
        # Email body
        html_body = self._generate_html_body(success, output_dir, timing_report, error_message)
        msg.attach(MIMEText(html_body, 'html'))
        
        return msg
    
    def _generate_html_body(
        self,
        success: bool,
        output_dir: Path,
        timing_report: str,
        error_message: Optional[str]
    ) -> str:
        """Generate HTML email body"""
        
        # Status banner
        if success:
            status_color = "#28a745"
            status_icon = "✅"
            status_text = "Pipeline Completed Successfully"
        else:
            status_color = "#dc3545"
            status_icon = "❌"
            status_text = "Pipeline Failed"
        
        # Build HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {status_color}; color: white; padding: 20px; border-radius: 5px; text-align: center; }}
                .content {{ background-color: #f9f9f9; padding: 20px; margin-top: 20px; border-radius: 5px; }}
                .section {{ margin-bottom: 20px; }}
                .label {{ font-weight: bold; color: #555; }}
                .timing {{ background-color: white; padding: 15px; font-family: monospace; font-size: 12px; border-left: 4px solid #007bff; white-space: pre-wrap; }}
                .error {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin-top: 20px; }}
                .footer {{ text-align: center; color: #777; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }}
                ul {{ list-style-type: none; padding-left: 0; }}
                li {{ padding: 5px 0; }}
                li:before {{ content: "📄 "; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{status_icon} {status_text}</h1>
                </div>
                
                <div class="content">
                    <div class="section">
                        <p class="label">Pipeline Information:</p>
                        <ul>
                            <li><strong>Server:</strong> {socket.gethostname()}</li>
                            <li><strong>Output Directory:</strong> {output_dir.absolute()}</li>
                        </ul>
                    </div>
        """
        
        # Add error section if failed
        if not success and error_message:
            html += f"""
                    <div class="error">
                        <p class="label">Error Details:</p>
                        <pre>{error_message}</pre>
                    </div>
            """
        
        # Add timing report
        html += f"""
                    <div class="section">
                        <p class="label">Timing Report:</p>
                        <div class="timing">{timing_report}</div>
                    </div>
        """
        
        # Add attachments note
        if success:
            html += """
                    <div class="section">
                        <p class="label">📎 Attachments:</p>
                        <p>The main Excel report is attached to this email. Additional detailed files are available in the output directory.</p>
                    </div>
            """
        
        # Footer
        html += f"""
                </div>
                
                <div class="footer">
                    <p>This is an automated message from the Donations Analytics Pipeline.</p>
                    <p>For questions or issues, please contact your system administrator.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _attach_files(
        self,
        msg: MIMEMultipart,
        output_dir: Path,
        success: bool
    ) -> None:
        """Attach output files to email"""
        
        if not success:
            return  # Don't attach files if pipeline failed
        
        # Find Excel file
        excel_files = list(output_dir.glob("*.xlsx"))
        
        if excel_files:
            excel_file = excel_files[0]  # Take the first one
            
            try:
                with open(excel_file, 'rb') as f:
                    part = MIMEBase('application', 'vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename={excel_file.name}'
                    )
                    msg.attach(part)
                    
                self.logger.info(f"Attached file: {excel_file.name}")
                
            except Exception as e:
                self.logger.warning(f"Could not attach {excel_file.name}: {e}")
        
        # Optionally attach timing report
        timing_file = output_dir.parent / "pipeline_timing.txt"
        if timing_file.exists():
            try:
                with open(timing_file, 'rb') as f:
                    part = MIMEBase('text', 'plain')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename={timing_file.name}'
                    )
                    msg.attach(part)
                    
                self.logger.debug(f"Attached file: {timing_file.name}")
                
            except Exception as e:
                self.logger.warning(f"Could not attach {timing_file.name}: {e}")
    
    def _send_email(self, msg: MIMEMultipart) -> None:
        """Send email via SMTP"""
        
        # Get all recipients
        recipients = self.config.email.to_addresses.copy()
        if self.config.email.cc_addresses:
            recipients.extend(self.config.email.cc_addresses)
        
        # Connect to SMTP server
        if self.config.email.use_tls:
            server = smtplib.SMTP(self.config.email.smtp_server, self.config.email.smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP(self.config.email.smtp_server, self.config.email.smtp_port)
        
        # Login if credentials provided
        if self.config.email.smtp_username and self.config.email.smtp_password:
            server.login(self.config.email.smtp_username, self.config.email.smtp_password)
        
        # Send email
        server.sendmail(
            self.config.email.from_address,
            recipients,
            msg.as_string()
        )
        
        server.quit()