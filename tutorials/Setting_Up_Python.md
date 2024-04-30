### Setting up to code in Python

You have now sucessfully installed VS Code to your machine.

In order to code in Python within VS Code, there are a few extra steps you need to complete.
    
The first step is to make sure you have Python installed on your machine.
    
*What It Is:* Python is a programming language. It's a way for people to communicate their instructions to a computer. You can write programs in Python to perform all sorts of tasks, from simple calculations to complex data analysis and web applications.
    
*Imagine It Like:* Python is like a special language you speak with your computer. You tell the computer what to do, step by step, using Python instructions. For example, you might tell the computer to add two numbers together, and Python helps you do that.
    
ThoughtSpace is written using the Python programming language.
    
If you already have Python installed, you can skip this step.
    
If you are unsure whether it is already installed, open up the "Command Prompt" from the start menu and type:

```
python --version
```

OR

```
python3 --version
```

Press enter.
    
If Python is installed, this command will return the installed Python version number. 
    
If Python is not installed, you will likely see an error message indicating that the command is not recognized. In that case, you would need to install Python on your machine using the steps described below.

**1. Install Python**

- If it is not on your machine, download and install the latest version of Python for Windows  from the official website (yellow box): https://www.python.org/downloads/

- You can opt for a custom installation, but the default "Install Now" option is fine.

- **IMPORTANT:** Make sure to check the box that says "Add Python to PATH." This option is important for Visual Studio Code to detect the Python installation.

- Selecting "Use admin privileges when installing py.exe" is also a good idea.

- Select "Yes" if it asks you whether you want to allow this app to make changes to your device.

- Select "Close" when it is finished.
    
You now have a Python installation on your machine.

**2. Install Python Extension in Visual Studio Code.**
    
You now need to install the Python extension in VS code to be able to use Python from within VS code.

- In Visual Studio Code, go to the Extensions view by clicking on the square icon in the sidebar on the left or by pressing Ctrl+Shift+X.

- Search for "Python" in the Extensions marketplace search bar.

- Install the one published by Microsoft (it should be the first result and the one with the blue tick). Click "Install" to install the extension.

- Note: if you have windows 11 and run into "Error while fetching extensions. XHR failed", the following link discusses a range of common fixes (https://stackoverflow.com/questions/70177216/visual-studio-code-error-while-fetching-extensions-xhr-failed)
and the following steps are a good place to start:


    - Open Windows Security
    - Click on Firewall & Network Security
    - Click on Allow an app through the firewall
    - Click on Change Settings
    - Click on Allow Another App
    - Browse to where VS Code is installed and click on Code.exe
    - Make sure that Code shows in the list of the allowable apps and that 'private' and 'public' networks are selected.


- Another easy fix to try: navigate to User settings using "Ctr+Shift+P", search "proxy", scroll down to "Proxy Support". The default is "override". Try switching to "off", restart visual studio and try again.

- If these suggestions don't fix it and for any other issues while setting up, stackoverflow often has good fixes and Chat GPT can also be useful here. 
    - If in doubt, chat to someone with more experience before making any big changes to your machine. This is particulary important if it is a uni-managed device.

**3. Select Python Interpreter**
    
You now need to select a Python Interpreter.
    
 *What It Is:* The Python interpreter is like a translator between human-readable Python code and the language that your computer understands. When you write Python code in VS Code, the Python interpreter takes that code and executes it, making your instructions come to life on your computer.

*Imagine It Like:* The Python interpreter is like a magical friend who understands both your language (Python) and your computer's language. You tell your magical friend what you want to do in Python, and they make sure your computer understands and follows your instructions.

- You can select the interpreter by opening the Command Palette (Ctrl+Shift+P), typing "Python: Select Interpreter," and choosing the interpreter from the list of detected Python interpreters.

**4. Verify it's all worked**

- Select "File", then "New File". You can then select "Python File" (ending with .py)

- Type the following in the first line: 

```
print ("Hello World")
```

- "Save as" this file to your computer (anywhere is fine as this is just a test).

- Then, in the top right corner select the arrow button, which says "Run Python File" when you hover over it.

- Make sure you have selected "Terminal" in the bottom window, and you should be able to see the print out "Hello World". 
    
- It should look something like this:
![image.png](https://hackmd.io/_uploads/rJoXGMeQ6.png)

- If you run into issues running this script, stackoverflow often has good fixes (be specific and concise in your search) and Chat GPT can also be useful here. 
- If in doubt, chat to someone with more experience before making any big changes to your machine.

### Recap of what you have done so far...

- Installed Python (and Python extension within visual studio code):
  - What It Is: Python is a programming language. It's a way for people to communicate their instructions to a computer. You can write programs in Python to perform all sorts of tasks, from simple calculations to complex data analysis and web applications.
  - Imagine It Like: Python is like a special language you speak with your computer. You tell the computer what to do, step by step, using Python instructions. For example, you might tell the computer to add two numbers together, and Python helps you do that.

- Selected Python Interpreter: 
  - What It Is: The Python interpreter is like a translator between human-readable Python code and the language that your computer understands. When you write Python code in VS Code, the Python interpreter takes that code and executes it, making your instructions come to life on your computer.
  - Imagine It Like: The Python interpreter is like a magical friend who understands both your language (Python) and your computer's language. You tell your magical friend what you want to do in Python, and they make sure your computer understands and follows your instructions.

#### Putting It All Together:

- So when you created your program that says "Hello, World!" on your computer screen, you wrote the instructions for this in Python (your language) inside Visual Studio Code (your notebook). When you ran your program, Visual Studio Code used the Python interpreter to translate your Python instructions into a language your computer understands. As a result, your computer displayed "Hello, World!" on the screen.
  
- In summary, Visual Studio Code is your workspace, Python is the language you use to communicate with your computer, and the Python interpreter is the translator that makes sure your computer understands and carries out your Python instructions.