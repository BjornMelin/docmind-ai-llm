import json
import csv
from collections import defaultdict
from pathlib import Path

# Files already processed (exclude from output)
PROCESSED_FILES = {'router_factory', '01_chat', 'coordinator'}

def should_exclude(file_path):
    """Check if file is already processed"""
    return any(pattern in file_path for pattern in PROCESSED_FILES)

def extract_threads(json_file):
    """Parse JSON and extract unresolved threads"""
    with open(json_file) as f:
        data = json.load(f)
    
    return data.get('unresolved_review_threads', [])

def get_code_block_indicator(thread):
    """Check if thread has suggested code"""
    for comment in thread.get('comments', []):
        body = comment.get('body', '')
        if '```' in body:
            return 'Yes'
    return 'No'

def get_issue_summary(thread):
    """Extract first comment as summary, limit to 80 chars"""
    comments = thread.get('comments', [])
    if comments:
        body = comments[0].get('body', '').split('\n')[0]
        # Remove markdown code blocks and extra spaces
        body = body.replace('```', '').strip()
        if body:
            return body[:80]
    return 'No comment'

def get_line_number(location):
    """Extract line number from location"""
    if isinstance(location, dict):
        return location.get('start_line', 'N/A')
    return 'N/A'

def main():
    json_file = Path('/home/bjorn/.claude/projects/-home-bjorn-repos-agents-docmind-ai-llm/8229b24c-f815-4512-984c-cd05b58d0f20/tool-results/toolu_01M2DwumbVTpBWKcrZcBnKgG.txt')
    
    if not json_file.exists():
        print(f"Error: File not found at {json_file}")
        return
    
    threads = extract_threads(json_file)
    print(f"Total unresolved threads in data: {len(threads)}\n")
    
    # Group by file and filter
    by_file = defaultdict(list)
    skipped_processed = 0
    
    for thread in threads:
        file_path = thread.get('file', 'unknown')
        
        # Skip processed files
        if should_exclude(file_path):
            skipped_processed += 1
            continue
        
        line_number = get_line_number(thread.get('location'))
        summary = get_issue_summary(thread)
        has_code = get_code_block_indicator(thread)
        
        by_file[file_path].append({
            'line': line_number,
            'summary': summary,
            'has_code': has_code,
            'thread_id': thread.get('thread_id'),
            'comment_count': thread.get('comment_count', 0),
        })
    
    print(f"Skipped (already processed): {skipped_processed}")
    print(f"Unresolved threads to handle: {sum(len(v) for v in by_file.values())}\n")
    
    # Print summary by file
    print("=" * 120)
    print("UNRESOLVED REVIEW THREADS (excluding router_factory, 01_chat, coordinator)")
    print("=" * 120)
    print()
    
    # Sort by file name
    for file_path in sorted(by_file.keys()):
        issues = by_file[file_path]
        print(f"\n{file_path}")
        print(f"  Count: {len(issues)} threads")
        print("-" * 120)
        
        # Sort by line number
        for issue in sorted(issues, key=lambda x: x['line'] if isinstance(x['line'], int) else 999999):
            line = issue['line']
            summary = issue['summary']
            has_code = issue['has_code']
            comments = issue['comment_count']
            print(f"  Line {str(line):6} | Code: {has_code:3} | Comments: {comments} | {summary}")
        
    # Print CSV
    print("\n\n" + "=" * 120)
    print("CSV OUTPUT (for spreadsheet import)")
    print("=" * 120 + "\n")
    
    output_csv = '/tmp/unresolved_reviews.csv'
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Line', 'Has Code Block', 'Comments', 'Issue Summary'])
        
        for file_path in sorted(by_file.keys()):
            for issue in sorted(by_file[file_path], key=lambda x: x['line'] if isinstance(x['line'], int) else 999999):
                writer.writerow([
                    file_path,
                    issue['line'],
                    issue['has_code'],
                    issue['comment_count'],
                    issue['summary'],
                ])
    
    # Print CSV content
    with open(output_csv) as f:
        print(f.read())
    
    # Summary stats
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    total_threads = sum(len(issues) for issues in by_file.values())
    total_with_code = sum(
        1 for issues in by_file.values() 
        for issue in issues 
        if issue['has_code'] == 'Yes'
    )
    print(f"Total unresolved threads (excl. processed): {total_threads}")
    print(f"Threads with code suggestions: {total_with_code} ({100*total_with_code/total_threads:.0f}%)")
    print(f"Threads without code: {total_threads - total_with_code} ({100*(total_threads - total_with_code)/total_threads:.0f}%)")
    print(f"Files with issues: {len(by_file)}")
    
    # Files with most threads (good targets for fixing)
    print(f"\nTop files by thread count:")
    sorted_files = sorted(by_file.items(), key=lambda x: len(x[1]), reverse=True)
    for file_path, issues in sorted_files[:10]:
        code_count = sum(1 for i in issues if i['has_code'] == 'Yes')
        print(f"  {len(issues):2} threads | {code_count:2} with code | {file_path}")
    
    print(f"\nCSV saved to: {output_csv}")

if __name__ == '__main__':
    main()
