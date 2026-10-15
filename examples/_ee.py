"""Explicit credential selection for examples; no authentication on import."""

import ee


def initialize(project, credentials_source):
    if credentials_source == "adc":
        import google.auth
        credentials, _ = google.auth.default(scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ])
        ee.Initialize(credentials=credentials, project=project)
    else:
        ee.Initialize(project=project)


def add_credentials_argument(parser):
    parser.add_argument("--credentials", choices=("ee", "adc"), default="ee",
                        help="ee: earthengine authenticate; adc: application-default credentials")
