from calculateur import *
from decideur import *
from comestible import *
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, Response, status
from fastapi import Path as pth

app = FastAPI()

@app.post("/models")
async def create_model(max_depth : int | None, csv : UploadFile = File(...)):
    try :
        if csv.filename == None:
            return Response(
                        content="Aucun DataSet Fournit",
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT
                )
        
        rapport = entrainer(Path(csv.filename))
        
        return Response(
            content=rapport,
            status_code=status.HTTP_201_CREATED
        )
    except DatasetInvalide :
        return Response(
            content="Le DataSet fournit ne permet pas d'entrainer un modèle",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT
        )
        
@app.get("/models")
def get_models():
    path_to_dir = Path("data")
    
    def is_csv_file(p : Path) -> bool:
        return p.is_file() and p.suffix == ".csv"
        
    result : list[Path] = list(filter(is_csv_file, path_to_dir.iterdir()))
    return Response(
        content=result,
        status_code=status.HTTP_200_OK
    )
    
@app.get("/models/{name}/features")
def get_features_of_model(name : Annotated[str, pth(title="Le nom du modèle à regarder")]):
    path_to_file = Path(f"data/{name}")
    if not path_to_file.is_file():
        return Response(
            content="Le modèle n'existe pas",
            status_code=status.HTTP_404_NOT_FOUND
        )
    