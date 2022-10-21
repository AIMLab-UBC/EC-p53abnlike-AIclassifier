echo """# AttentionMIL

### Development Information ###
\`\`\`
Date Created: 20 Aug 2022
Developer: Amirali
Version: 0.0
\`\`\`

### About The Project ###
This repo contains the implementation of "*Artificial intelligence-based histopathology image analysis identifies a novel subset of endometrial cancers with unfavourable outcome*". The below GIF illustrates the proposed workflow.

![](gif/workflow.gif)


## Installation

\`\`\`
mkdir AttentionMIL
cd AttentionMIL
git clone git clone https://svn.bcgsc.ca/bitbucket/scm/~adarbandsari/attentionmil.git .
pip install -r requirements.txt
\`\`\`


### Usage ###
\`\`\`
""" > README.md

python run.py -h >> README.md
echo >> README.md
python run.py calculate-representation -h >> README.md
echo >> README.md
python run.py train-attention -h >> README.md
echo >> README.md
python run.py train-attention VarMIL -h >> README.md
echo >> README.md

echo """\`\`\`
""" >> README.md
