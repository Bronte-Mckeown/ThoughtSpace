## Doing this all again following updates to ThoughtSpace...
    
As ThoughtSpace is still in development, you may want to update your version of ThoughtSpace as changes are made.
    
If you have forked and then cloned, do the following first:

1. Go to your forked repository on GitHub.
2. Click on the "Fetch upstream" button
    - This button is usually found on the top right side of the repository page, near the "Code" button.
3. Click on the "Create pull request" button. This will take you to a new pull request page.
4. GitHub will automatically set the base repository and base branch to your fork and your default branch (e.g., main or master). It will also set the head repository and branch to the original repository and the branch from which you want to merge changes.
5. Review the changes and click on the "Create pull request" button to create the pull request. Add a title and description if necessary, and then click "Create pull request" again.
6. Finally, click on "Merge pull request" to merge the changes from the original repository into your forked repository.

Your forked repository is now synced with the changes from the original repository.

If you skipped forking and just simply cloned ThoughtSpace directly, do the following first:
    
1. Open GitHub desktop
2. Select the ThoughtSpace repository
3. Select "Fetch origin". This will fetch the changes from the remote ThoughtSpace repository and update your local copy of this repository.

Now that your local copy of ThoughtSpace is up-to-date, do the following:
    1. Open the Command Prompt
    2. Activate your conda environment as before (see above)
    3. Navigate to ThoughtSpace directory using 'cd' as before (see above)
    4. Type "pip install .", press Enter.

This step will uninstall the old version of ThoughtSpace and re-install the new version.