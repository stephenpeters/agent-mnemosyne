#!/usr/bin/env python3
"""
Email Notification System for Mnemosyne

Sends email notifications for:
- Pipeline run statistics
- System errors and failures
- Health check alerts
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send email notifications via Fastmail SMTP"""

    def __init__(self):
        self.enabled = self._check_config()
        if self.enabled:
            self.smtp_server = "smtp.fastmail.com"
            self.smtp_port = 587
            self.from_addr = os.getenv("FASTMAIL_ADDRESS", "stephenp@fastmail.com.au")
            self.password = os.getenv("FASTMAIL_PASSWORD")
            self.to_addr = "mnemosyne@psc.net.au"
            logger.info(f"Email notifications enabled: {self.from_addr} → {self.to_addr}")
        else:
            logger.warning("Email notifications disabled - missing credentials")

    def _check_config(self) -> bool:
        """Check if email credentials are configured"""
        return bool(os.getenv("FASTMAIL_ADDRESS") and os.getenv("FASTMAIL_PASSWORD"))

    def send_pipeline_stats(self, stats: Dict):
        """
        Send pipeline run statistics

        Args:
            stats: Pipeline statistics with successes, failures, metrics
        """
        if not self.enabled:
            return

        try:
            subject = f"✅ Mnemosyne Pipeline Complete - {stats.get('successes', 0)} ideas processed"

            body = self._format_pipeline_email(stats)

            self._send_email(subject, body)
            logger.info("Sent pipeline stats email")

        except Exception as e:
            logger.error(f"Failed to send pipeline stats email: {e}")

    def send_error_notification(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """
        Send error notification

        Args:
            error_type: Type of error (e.g., "Pipeline Failure", "Agent Down")
            error_message: Error message
            context: Additional context information
        """
        if not self.enabled:
            return

        try:
            subject = f"🚨 Mnemosyne Error: {error_type}"

            body = self._format_error_email(error_type, error_message, context)

            self._send_email(subject, body)
            logger.info(f"Sent error notification: {error_type}")

        except Exception as e:
            logger.error(f"Failed to send error email: {e}")

    def send_health_alert(self, health_score: int, issues: List[str]):
        """
        Send health alert when system health is degraded

        Args:
            health_score: System health score (0-100)
            issues: List of detected issues
        """
        if not self.enabled:
            return

        # Only send if health is below threshold
        if health_score >= 80:
            return

        try:
            status = "⚠️  DEGRADED" if health_score >= 60 else "🚨 CRITICAL"
            subject = f"{status} Mnemosyne Health Alert - Score: {health_score}/100"

            body = self._format_health_email(health_score, issues)

            self._send_email(subject, body)
            logger.info(f"Sent health alert: {health_score}/100")

        except Exception as e:
            logger.error(f"Failed to send health alert: {e}")

    def _format_pipeline_email(self, stats: Dict) -> str:
        """Format pipeline statistics as HTML email"""

        run_id = stats.get('run_id', 'N/A')
        started = stats.get('started_at', 'N/A')
        completed = stats.get('completed_at', 'N/A')
        successes = stats.get('successes', 0)
        failures = stats.get('failures', 0)
        total = successes + failures

        metrics = stats.get('metrics', {})
        avg_ai = metrics.get('avg_ai_likelihood', 0)
        avg_voice = metrics.get('avg_voice_deviation', 0)
        quality_passed = metrics.get('quality_passed', 0)
        quality_failed = metrics.get('quality_failed', 0)

        # Success rate
        success_rate = (successes / total * 100) if total > 0 else 0

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #28a745; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stats {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .stat-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #dee2e6; }}
        .stat-label {{ font-weight: bold; }}
        .success {{ color: #28a745; font-weight: bold; }}
        .failure {{ color: #dc3545; font-weight: bold; }}
        .metric {{ background: white; padding: 15px; margin: 10px 0; border-left: 3px solid #007bff; }}
        .footer {{ text-align: center; color: #6c757d; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✅ Pipeline Complete</h1>
            <p>Mnemosyne Content Pipeline</p>
        </div>

        <div class="stats">
            <h2>Run Summary</h2>
            <div class="stat-row">
                <span class="stat-label">Run ID:</span>
                <span>{run_id}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Started:</span>
                <span>{started[:19] if len(started) > 19 else started}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Completed:</span>
                <span>{completed[:19] if len(completed) > 19 else completed}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Success Rate:</span>
                <span class="success">{success_rate:.1f}%</span>
            </div>
        </div>

        <div class="stats">
            <h2>Processing Results</h2>
            <div class="stat-row">
                <span class="stat-label">Total Ideas:</span>
                <span>{total}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Successful:</span>
                <span class="success">{successes}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Failed:</span>
                <span class="failure">{failures}</span>
            </div>
        </div>

        <div class="stats">
            <h2>Quality Metrics</h2>
            <div class="metric">
                <strong>Average AI Likelihood:</strong> {avg_ai:.3f}<br>
                <small>Target: &lt; 0.25</small>
            </div>
            <div class="metric">
                <strong>Average Voice Deviation:</strong> {avg_voice:.3f}<br>
                <small>Target: &lt; 0.35</small>
            </div>
            <div class="metric">
                <strong>Quality Checks:</strong><br>
                <span class="success">✓ Passed: {quality_passed}</span><br>
                <span class="failure">✗ Failed: {quality_failed}</span>
            </div>
        </div>

        <div class="footer">
            <p>Mnemosyne - Autonomous Content Pipeline</p>
            <p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _format_error_email(self, error_type: str, error_message: str, context: Optional[Dict]) -> str:
        """Format error notification as HTML email"""

        context_html = ""
        if context:
            context_html = "<div class='context'><h3>Context</h3><ul>"
            for key, value in context.items():
                context_html += f"<li><strong>{key}:</strong> {value}</li>"
            context_html += "</ul></div>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #dc3545; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .error {{ background: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .error-type {{ font-size: 24px; font-weight: bold; color: #721c24; margin-bottom: 10px; }}
        .error-message {{ background: white; padding: 15px; border-left: 3px solid #dc3545; font-family: monospace; }}
        .context {{ background: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 8px; }}
        .context ul {{ margin: 10px 0; padding-left: 20px; }}
        .footer {{ text-align: center; color: #6c757d; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 System Error</h1>
            <p>Mnemosyne Alert</p>
        </div>

        <div class="error">
            <div class="error-type">{error_type}</div>
            <div class="error-message">{error_message}</div>
        </div>

        {context_html}

        <div class="footer">
            <p>Mnemosyne - Autonomous Content Pipeline</p>
            <p>Alert generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _format_health_email(self, health_score: int, issues: List[str]) -> str:
        """Format health alert as HTML email"""

        status_color = "#dc3545" if health_score < 60 else "#ffc107"
        status_text = "CRITICAL" if health_score < 60 else "DEGRADED"

        issues_html = "<ul>"
        for issue in issues:
            issues_html += f"<li>{issue}</li>"
        issues_html += "</ul>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: {status_color}; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .health-score {{ font-size: 48px; font-weight: bold; margin: 20px 0; }}
        .status {{ font-size: 24px; margin-bottom: 10px; }}
        .issues {{ background: #fff3cd; border-left: 3px solid {status_color}; padding: 20px; margin: 20px 0; }}
        .footer {{ text-align: center; color: #6c757d; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="status">{status_text}</div>
            <h1>System Health Alert</h1>
            <div class="health-score">{health_score}/100</div>
        </div>

        <div class="issues">
            <h2>Detected Issues</h2>
            {issues_html}
        </div>

        <div class="footer">
            <p>Mnemosyne - Autonomous Content Pipeline</p>
            <p>Health check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _send_email(self, subject: str, html_body: str):
        """Send HTML email via SMTP"""

        msg = MIMEMultipart('alternative')
        msg['From'] = self.from_addr
        msg['To'] = self.to_addr
        msg['Subject'] = subject

        # Attach HTML body
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)

        # Send via SMTP
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.from_addr, self.password)
            server.send_message(msg)


# Global instance
notifier = EmailNotifier()
