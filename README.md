# Repository usage guide for students

This repository will be used during the course/project.  
The instructions below summarize the basic Git workflow.

## General rule

Each student must work on their **own branch**.

Please **do not commit directly to `main`**.

---

## 1. First-time Git setup

If Git is not yet configured on your computer, run:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
````

You only need to do this once.

---

## 2. Clone the repository on your computer

```bash
git clone <repository-url>
cd <repository-folder>
```

---

## 3. Create your own branch

After cloning the repository, create a branch with your name and the project you are working on.

Example:

```bash
git checkout -b student-name/proj-01
```

This command creates your branch and switches you to it.

---

## 4. Check which branch you are using

```bash
git branch
```

The current branch appears with `*`.

Example:

```bash
* bruno/proj-01
  main
```

Always make sure you are on **your own branch** before making changes.

---

## 5. Basic workflow

Whenever you start working, follow this sequence:

### Update your local repository

```bash
git pull
```

### Check the repository status

```bash
git status
```

### Add your changes

To add all modified files:

```bash
git add .
```

Or, to add only one file:

```bash
git add filename
```

### Commit your changes

```bash
git commit -m "Short description of your changes"
```

### General instruction for commits

Try to make commits that are clear and focused.
A commit message should briefly describe what you changed, for example:

* `Fix plotting script`
* `refactor ...`
* `Add results`

### Push your branch to GitHub

The first time you push your branch, use:

```bash
git push -u origin your-branch-name
```

After the first push, you can simply use:

```bash
git push
```

---

## 6. Recommended day-to-day workflow

After your branch has already been created, the usual workflow is:

```bash
git checkout your-branch-name
git pull
git status
git add .
git commit -m "Describe your changes"
git push
```

---

## 7. Useful commands

### See the current status

```bash
git status
```

### See the commit history

```bash
git log --oneline
```

### See file differences

```bash
git diff
```

### Restore a file to the last committed version

```bash
git restore filename
```
