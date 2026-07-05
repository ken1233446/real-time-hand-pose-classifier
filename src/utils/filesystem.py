from pathlib import Path

class FileSystem:
    
    #Handle creating Poses folder.
    #Folder creation. | Return: text
    def create_folder(self, folder_path: Path, text: str) -> str:
    
        #Create folder for dataset.
        try: 
            folder_path.mkdir(parents=True)
            print(f"New Pose folder created at {folder_path} named {text}")

        #Check if the folder already exist.
        except FileExistsError:
            print(f"Folder already exists: {folder_path}")
        
        #Check for any Error in creating folder.
        except OSError as e:
            print(f"Error creating folder: {e}")

        #Update the text to default empty after pressing enter.
        text = ""

        return text
