import gmsh
from numpy import array, int32, concatenate

def get_physical_groups_map(gmshmodel):
	"""
	Given the gmsh model, return a map of all defined physical groups and their names.
	The map will return the dimension and rag of the physical group if indexed by name
	"""
	pg = gmshmodel.getPhysicalGroups()
	the_physical_groups_map = {}
	for dim, tag in pg:
		name = gmshmodel.getPhysicalName(dim, tag)
		the_physical_groups_map[name] = (dim, tag)

	return the_physical_groups_map


def get_elements_and_nodes_in_physical_group(groupname, gmshmodel):
	"""
	Returns element tags, node tags (connectivity), element name (gmsh element type name), and
	number of nodes for the element type, given the name of a physical group. Inputs are the physical
	group string name and the gmsh model 
	"""

	dim, tag  = get_physical_groups_map(gmshmodel)[groupname]  
	entities = gmshmodel.getEntitiesForPhysicalGroup(dim, tag)


	allelementtags = array([], dtype=int32)
	allnodetags = array([], dtype=int32)

	base_element_type = -1

	for e in entities:
		elementTypes, elementTags, nodeags = gmshmodel.mesh.getElements(dim, e)

		if len(elementTypes) != 1:
			print("Cannot handle more than one element type at this moment. Contributions welcome. ")
			exit(-1)

		if base_element_type == -1:
			base_element_type = elementTypes[0]
		elif elementTypes[0] != base_element_type:
			print("All entities of physical group should have the same element type. Contributions welcome. ")
			exit(-1)


		allelementtags = concatenate((allelementtags,elementTags[0]))
		allnodetags = concatenate((allnodetags,nodeags[0]))

	element_name, element_nnodes = get_element_info_from_elementType(base_element_type)
	allnodetags = allnodetags.reshape((-1,element_nnodes))

		
	return int32(allelementtags).tolist(), int32(allnodetags).tolist(), element_name, element_nnodes

def get_element_info_from_elementType(elementType):
	"""
	Returns element gmsh name and number of nodes given element type
	Can be extended to add other elements.
	"""
	info = {
	#  elementType    Name                  Number of nodes
		1         : ( "2-node-line"         , 2       )  ,
		2         : ( "3-node-triangle"     , 3       )  ,
		3         : ( "4-node-quadrangle"   , 4       )  ,
		4         : ( "4-node-tetrahedron"  , 4       )  ,
		5         : ( "8-node-hexahedron"   , 8       )  ,
  		8		  : ( "3-node-line"         , 3       )  ,
		9         : ( "6-node-triangle"     , 6       )  ,
		10		  : ( "9-node-quadrilateral" , 9 	  )  ,
		11        : ( "10-node-tetrahedron" , 10      )  ,
		15        : ( "1-node-point"        , 1       )  ,
	}
	if elementType in info:
		return info[elementType]
	else:
		print(f"elementType={elementType} unavailable. Contributions welcome. See https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format")
		exit(-1)
