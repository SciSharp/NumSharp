#!/usr/bin/env python3
"""
Track GitHub repo stargazers and watchers, detect changes over time.

Usage:
    python track_repo_followers.py [--repo OWNER/REPO] [--token TOKEN]
    python track_repo_followers.py --history          # Show star history by date
    python track_repo_followers.py --history --days 30  # Last 30 days

Environment:
    GITHUB_TOKEN or GH_TOKEN - GitHub personal access token (optional but recommended for rate limits)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DEFAULT_REPO = "SciSharp/NumSharp"
STATE_FILE = Path(__file__).parent / "repo_followers_state.json"


def get_token():
    """Get GitHub token from environment."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def github_api(endpoint: str, token: str = None, accept: str = None) -> dict | list:
    """Make a GitHub API request."""
    url = f"https://api.github.com{endpoint}"
    headers = {
        "Accept": accept or "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_paginated(endpoint: str, token: str = None, per_page: int = 100, accept: str = None) -> list:
    """Fetch all pages from a paginated GitHub API endpoint."""
    results = []
    page = 1
    while True:
        sep = "&" if "?" in endpoint else "?"
        data = github_api(f"{endpoint}{sep}per_page={per_page}&page={page}", token, accept)
        if not data:
            break
        results.extend(data)
        if len(data) < per_page:
            break
        page += 1
    return results


def get_stargazers_with_dates(repo: str, token: str = None) -> dict[str, str]:
    """Get all stargazers with their starred_at timestamps."""
    # Use special Accept header to get timestamps
    users = get_paginated(
        f"/repos/{repo}/stargazers",
        token,
        accept="application/vnd.github.star+json"
    )
    # Returns {login: starred_at}
    return {u["user"]["login"]: u["starred_at"] for u in users}


def get_stargazers(repo: str, token: str = None) -> list[str]:
    """Get all stargazers for a repo (names only)."""
    stargazers = get_stargazers_with_dates(repo, token)
    return sorted(stargazers.keys(), key=str.lower)


def get_watchers(repo: str, token: str = None) -> list[str]:
    """Get all watchers (subscribers) for a repo."""
    users = get_paginated(f"/repos/{repo}/subscribers", token)
    return sorted([u["login"] for u in users], key=str.lower)


def load_state() -> dict:
    """Load previous state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save state to file."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def detect_changes(old_list: list[str], new_list: list[str]) -> tuple[list[str], list[str]]:
    """Detect additions and removals between two lists."""
    old_set = set(old_list)
    new_set = set(new_list)
    added = sorted(new_set - old_set, key=str.lower)
    removed = sorted(old_set - new_set, key=str.lower)
    return added, removed


def show_history(repo: str, token: str, days: int = None, as_json: bool = False):
    """Show star history grouped by date."""
    print(f"Fetching stargazer history for {repo}...", file=sys.stderr)

    stargazers = get_stargazers_with_dates(repo, token)

    # Group by date
    by_date = defaultdict(list)
    for user, starred_at in stargazers.items():
        date = starred_at[:10]  # YYYY-MM-DD
        by_date[date].append(user)

    # Filter by days if specified
    if days:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        by_date = {d: users for d, users in by_date.items() if d >= cutoff}

    # Sort by date descending
    sorted_dates = sorted(by_date.keys(), reverse=True)

    if as_json:
        result = {
            "repo": repo,
            "total_stars": len(stargazers),
            "history": {d: by_date[d] for d in sorted_dates},
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\n=== Star History: {repo} ===")
        print(f"Total stars: {len(stargazers)}")
        if days:
            print(f"Showing last {days} days\n")
        else:
            print()

        cumulative = len(stargazers)
        for date in sorted_dates:
            users = by_date[date]
            print(f"{date}: +{len(users):3d} (total: {cumulative:,d})  {', '.join(sorted(users, key=str.lower))}")
            cumulative -= len(users)

        if not sorted_dates:
            print("No stars in the specified period.")


def main():
    parser = argparse.ArgumentParser(description="Track GitHub repo stargazers and watchers")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Repository (default: {DEFAULT_REPO})")
    parser.add_argument("--token", help="GitHub token (or set GITHUB_TOKEN/GH_TOKEN env var)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--history", action="store_true", help="Show star history by date")
    parser.add_argument("--days", type=int, help="Limit history to last N days")
    args = parser.parse_args()

    token = args.token or get_token()
    repo = args.repo

    if not token:
        print("Warning: No GitHub token provided. Rate limits will be strict (60 req/hour).", file=sys.stderr)

    try:
        # History mode
        if args.history:
            show_history(repo, token, args.days, args.json)
            return

        print(f"Fetching data for {repo}...", file=sys.stderr)

        stargazers_with_dates = get_stargazers_with_dates(repo, token)
        stargazers = sorted(stargazers_with_dates.keys(), key=str.lower)
        watchers = get_watchers(repo, token)

        now = datetime.utcnow().isoformat() + "Z"

        # Load previous state
        state = load_state()
        prev = state.get(repo, {})

        # Detect changes
        stars_added, stars_removed = detect_changes(prev.get("stargazers", []), stargazers)
        watchers_added, watchers_removed = detect_changes(prev.get("watchers", []), watchers)

        # Build result with star dates
        prev_dates = prev.get("stargazers_dates", {})
        result = {
            "repo": repo,
            "timestamp": now,
            "previous_check": prev.get("timestamp"),
            "stargazers": {
                "count": len(stargazers),
                "previous_count": len(prev.get("stargazers", [])),
                "added": {u: stargazers_with_dates.get(u) for u in stars_added},
                "removed": {u: prev_dates.get(u) for u in stars_removed},
            },
            "watchers": {
                "count": len(watchers),
                "previous_count": len(prev.get("watchers", [])),
                "added": watchers_added,
                "removed": watchers_removed,
            },
        }

        # Save new state (include starred_at dates for history tracking)
        state[repo] = {
            "timestamp": now,
            "stargazers": stargazers,
            "stargazers_dates": stargazers_with_dates,  # {user: starred_at}
            "watchers": watchers,
        }
        save_state(state)

        # Output
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n=== {repo} ===")
            print(f"Checked: {now}")
            if prev.get("timestamp"):
                print(f"Previous: {prev['timestamp']}")

            print(f"\nStargazers: {len(stargazers)}", end="")
            if prev.get("stargazers"):
                diff = len(stargazers) - len(prev["stargazers"])
                if diff != 0:
                    print(f" ({'+' if diff > 0 else ''}{diff})", end="")
            print()

            if stars_added:
                # Show with star dates
                added_with_dates = []
                for user in stars_added:
                    date = stargazers_with_dates.get(user, "")[:10]
                    added_with_dates.append(f"{user} ({date})" if date else user)
                print(f"  + New: {', '.join(added_with_dates)}")
            if stars_removed:
                # Show with original star dates if we have them
                prev_dates = prev.get("stargazers_dates", {})
                removed_with_dates = []
                for user in stars_removed:
                    date = prev_dates.get(user, "")[:10] if prev_dates.get(user) else ""
                    removed_with_dates.append(f"{user} (starred {date})" if date else user)
                print(f"  - Lost: {', '.join(removed_with_dates)}")

            print(f"\nWatchers: {len(watchers)}", end="")
            if prev.get("watchers"):
                diff = len(watchers) - len(prev["watchers"])
                if diff != 0:
                    print(f" ({'+' if diff > 0 else ''}{diff})", end="")
            print()

            if watchers_added:
                print(f"  + New: {', '.join(watchers_added)}")
            if watchers_removed:
                print(f"  - Lost: {', '.join(watchers_removed)}")

            if not any([stars_added, stars_removed, watchers_added, watchers_removed]):
                if prev:
                    print("\nNo changes since last check.")
                else:
                    print("\nFirst run - baseline recorded.")

    except HTTPError as e:
        print(f"GitHub API error: {e.code} {e.reason}", file=sys.stderr)
        if e.code == 401:
            print("Invalid or expired token.", file=sys.stderr)
        elif e.code == 403:
            print("Rate limited or forbidden. Try providing a token.", file=sys.stderr)
        elif e.code == 404:
            print(f"Repository '{repo}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
