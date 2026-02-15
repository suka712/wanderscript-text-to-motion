# kuzen.quest

> Sprints that move themselves. Connect GitHub, plan your sprint, let git activity do the rest.

---

## User Flow

1. **Sign in** — GitHub OAuth → install GitHub App on org → select repos → done.
2. **Create sprint** — name, goal, 1-2 week duration.
3. **Fill sprint** — backlog pulls open issues from connected repos. Drag into sprint, assign points.
4. **Work** — git activity auto-moves cards. Branch created → In Progress. PR opened → In Review. PR merged → Done.
5. **Ship** — sprint closes, see what shipped, what carried over, velocity.

---

## MVP Features

### Board
- Four columns: **Backlog → In Progress → In Review → Done**
- Cards: title, assignee avatar, points, repo tag
- Manual drag override
- Filter by assignee, repo

### Automation (the killer feature)
- Branch created referencing issue → In Progress
- PR opened → In Review
- PR merged → Done
- That's it for v1. No custom rules yet.

### Sprint Management
- Create/close sprints with dates
- Backlog view — open issues across repos
- Drag issues into sprint, assign story points
- Carry-over unfinished issues to next sprint

### GitHub Sync
- One-way for MVP: GitHub → kuzen (webhooks + GraphQL)
- New issues auto-appear in backlog
- PR/branch activity drives card status

### Analytics (post-MVP, skip for now)
- Burndown chart
- Velocity over time
- Cycle time, PR turnaround
- Contributor breakdown

---

## Stack

| Layer | Choice |
|-------|--------|
| Frontend | React + TypeScript + Tailwind |
| Backend | Go |
| Database | Postgres |
| Cache | Redis |
| GitHub | GitHub App + GraphQL API + Webhooks |

---

## Build Order

1. GitHub OAuth + App install + repo selection
2. Issue sync (GraphQL pull + webhook listener)
3. Sprint CRUD + backlog
4. Board with drag
5. Webhook automation (branch/PR → status change)

**That's your MVP. Ship it.**

---

## Not Building (yet)

- Two-way sync (kuzen → GitHub)
- Analytics/charts
- Custom automation rules
- Comments
- Canvas/whiteboard planning
- Team management beyond GitHub org
- Any AI features