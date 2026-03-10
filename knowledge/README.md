# Knowledge Base Documents

Place your company documentation here to improve AI draft quality. Documents in this folder will be indexed and used to provide context when generating responses.

## Supported Formats

- Markdown (`.md`) - Recommended
- Plain text (`.txt`)
- PDF (`.pdf`) - Requires `pdf-parse` package

## File Organization

Organize by topic for better retrieval:

```
knowledge/
├── maintenance/
│   ├── emergency-procedures.md
│   ├── routine-maintenance.md
│   └── vendor-contacts.md
├── billing/
│   ├── payment-policies.md
│   ├── late-fees.md
│   └── refund-process.md
├── leasing/
│   ├── application-process.md
│   ├── move-in-checklist.md
│   └── lease-terms.md
└── general/
    ├── contact-info.md
    ├── office-hours.md
    └── faq.md
```

## Document Format

For best results, structure your documents with clear headings:

```markdown
# Emergency Maintenance Procedures

## What Qualifies as an Emergency

- Water leaks causing damage
- No heat when outside temperature is below 50°F
- No air conditioning when outside temperature is above 90°F
- Electrical hazards
- Security issues (broken locks, broken windows)
- Gas leaks (call 911 first!)

## After-Hours Emergency Line

Call: (555) 123-4567

Available 24/7 for true emergencies only.

## Non-Emergency Maintenance

For non-emergency issues, submit a maintenance request through the resident portal
or email maintenance@example.com. Requests are typically addressed within 48 hours.

## Common Questions

### Q: My toilet is running. Is this an emergency?
A: A running toilet is not an emergency unless it's overflowing. Submit a regular
maintenance request and we'll address it within 48 hours.
```

## Importing Documents

After adding or updating documents, run:

```bash
npm run import:docs
```

This will:
1. Read all documents in the `knowledge/` directory
2. Chunk them into smaller pieces for retrieval
3. Generate embeddings using OpenAI
4. Store in the vector database

## Tips for Good Documentation

1. **Be specific**: Include actual phone numbers, email addresses, and timelines
2. **Use Q&A format**: Common questions help the AI match to similar inquiries
3. **Include policies**: Reference actual policy numbers or lease sections
4. **Keep updated**: Re-run import after any changes
5. **Cover edge cases**: Document what to do in unusual situations

## Excluding Files

To exclude files from indexing, add them to a `.kbignore` file:

```
# .kbignore
drafts/
internal-only.md
*.tmp
```
