from PyMpc import *
from STKO_APE import modelAPE
from STKO_APE import write_in_terminal_blue, write_in_terminal_red, write_in_terminal

file=modelAPE()

print('---------------------------------------')
selection_set_id=3
ss_results=file.extract_mesh_data_selectionSet(selection_set_id=selection_set_id)

print(ss_results)

preprocessor=App.caeDocument()
mesh_object=preprocessor.mesh

def get_adjoining_faces(edges):

    adjoining_faces = {}
    
    # Iterate through the given edges
    for edge_id in edges:
        adjoining_faces[edge_id] = set()  # Initialize a set for adjoining faces
        
        # Search through all mesh geometries
        for geometry_id in mesh_object.meshedGeometries:
            mesh_geometry = mesh_object.meshedGeometries[geometry_id]
            
            # Check if the edge exists in the geometry
            if edge_id in [edge.id for edge in mesh_geometry.edges]:
                # Retrieve the edge object
                edge_obj = mesh_geometry.edges.__getitem__(edge_id)
                
                # Access faces connected to the edge
                for face in edge_obj.adjacentFaces:  # 'adjacentFaces' provides connected faces
                    adjoining_faces[edge_id].add(face.id)  # Add the face ID to the set
                    
    return adjoining_faces
    
aj=get_adjoining_faces([11])
