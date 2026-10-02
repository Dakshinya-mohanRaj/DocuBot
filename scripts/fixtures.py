"""Fixture corpus for the DocuBot multi-document retrieval evaluation.

Three documents with deliberately non-overlapping facts and distinct numeric
values, so that (a) ground truth is unambiguous and (b) an answer that leaks a
number from the wrong document is detectable by substring match.

Every number here is unique to exactly one document:
    1.5 / 30 / 2      -> handbook.txt
    100 / 30 / 25     -> spec.txt
    5 / 14 / 90       -> postmortem.txt
"""

HANDBOOK = """Northwind Engineering Employee Handbook, Revision 7.

Time off
Employees accrue 1.5 days of paid time off for each full month of service.
Accrual is capped at 30 days. Unused days above the cap are forfeited at the
end of the calendar year and cannot be paid out.

Remote work
Staff may work remotely 2 days per week by default. Managers may approve a
third remote day for employees with more than three years of tenure. Fully
remote arrangements require VP sign-off and are reviewed every six months.

Learning budget
Each employee receives 800 dollars per year for conferences, courses and
certifications. Requests are approved by your direct manager.

Equipment
Laptops are refreshed on a 36 month cycle. Employees may request a second
monitor at any time without additional approval.
"""

SPEC = """Northwind API — Product Specification, version 4.2.

Rate limiting
The API enforces a limit of 100 requests per minute for each API key. Keys
that exceed the limit receive HTTP 429 and are throttled for the remainder of
the minute. Rate limit headers are returned on every response.

Timeouts
Every outbound request has a timeout of 30 seconds. Clients that exceed this
window receive HTTP 504. Long-running jobs should be split into smaller
calls rather than extending the timeout.

File uploads
The maximum upload size for a single file is 25 MB. Larger files must be
split by the client before submission. Uploads above the limit are rejected
with HTTP 413.

Environments
Three environments are provided: sandbox, staging and production. Sandbox
data is reset nightly at 02:00 UTC.
"""

POSTMORTEM = """Incident Report: March API Outage, 2024-03-14.

Summary
On 2024-03-14 the production API was unavailable for 47 minutes. The outage
was caused by a retry storm after a single upstream provider returned slow
responses. Clients retried aggressively, amplifying load until the connection
pool was exhausted.

Detection
The on-call engineer was paged 8 minutes after the first error rate increase.
Alerting was delayed because the synthetic check had been disabled during a
routine migration earlier that week.

Resolution
A circuit breaker was added in front of the upstream provider. The retry
limit was reduced to 5 attempts with exponential backoff, and connection pool
limits were raised. Total time to full recovery was 90 minutes from the first
alert, including the 47 minutes of confirmed downtime.

Follow-up
Synthetic monitoring was re-enabled within one day. The team committed to
reviewing retry budgets for every outbound call.
"""


DOCUMENTS = {
    "handbook.txt": HANDBOOK,
    "spec.txt": SPEC,
    "postmortem.txt": POSTMORTEM,
}