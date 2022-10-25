# generates latex to list the feature plots
import glob


files = glob.glob("data/featurePlots/D4/*.png")
string = ""
hCount = 4
figWidth = 1/hCount * 0.9

for file in files:
    # Get file name
    fileName = file.split("\\")[-1]
    string += f"\\includegraphics[width=" + str(figWidth) + f"\\textwidth]{{featurePlots/D4/{fileName}}} & "

print(string)
