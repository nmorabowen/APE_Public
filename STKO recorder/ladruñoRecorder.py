# TCL generation module
import opensees.utils.tcl_input as tclin

# STKO integration module
import PyMpc.App
import PyMpc.Units as u
from PyMpc import *
from mpc_utils_html import *

# File handling modules
import csv
import importlib
import os

# ============================================
# GUI
# ============================================

def makeXObjectMetaData():
    """
    Function to create the metadata for the custom recorder object, allowing STKO to 
    display and configure user-defined attributes for the AlbertisRecorder.
    """
    def mka(type, name, group, descr):
        """ Helper function to streamline the creation of MpcAttributeMetaData objects. """
        a = MpcAttributeMetaData()
        a.type = type
        a.name = name
        a.group = group
        a.description = (
            html_par(html_begin()) +
            html_par(html_boldtext(name)+'<br/>') +
            html_par(descr) +
            html_par(html_href('https://opensees.berkeley.edu/wiki/index.php/Node_Recorder', 'Node Recorder')+'<br/>') +
            html_end()
        )
        return a

    # File Attributes Creation
    at_file_name = mka(MpcAttributeType.String, '$fileName', 'File', 'Name of file to which output is sent (without extension)')
    at_file = mka(MpcAttributeType.Boolean, '-file', 'File', 'Select if the output should be in a plain file format.')
    at_extension = mka(MpcAttributeType.String, 'Extension', 'File', 'Specify the file extension (default is .txt).')
    at_extension.setDefault('.txt')

    # DOF Attributes Creation
    at_dof_X = mka(MpcAttributeType.Boolean, '-dof_X', 'DOF', 'Track response for X direction.')
    at_dof_Y = mka(MpcAttributeType.Boolean, '-dof_Y', 'DOF', 'Track response for Y direction.')
    at_dof_Z = mka(MpcAttributeType.Boolean, '-dof_Z', 'DOF', 'Track response for Z direction.')

    # Node Attributes Creation
    at_nodes = mka(MpcAttributeType.IndexVector, '-node', 'Nodes', 'Tags of nodes whose response is being recorded.')
    at_nodes.indexSource.type = MpcAttributeIndexSourceType.SelectionSet

    # Response Type Attributes Creation
    at_disp = mka(MpcAttributeType.Boolean, 'disp', 'Response Type', 'Record displacement response.')
    at_vel = mka(MpcAttributeType.Boolean, 'vel', 'Response Type', 'Record velocity response.')
    at_accel = mka(MpcAttributeType.Boolean, 'accel', 'Response Type', 'Record acceleration response.')
    at_incrDisp = mka(MpcAttributeType.Boolean, 'incrDisp', 'Response Type', 'Record incremental displacement response.')
    at_reaction = mka(MpcAttributeType.Boolean, 'reaction', 'Response Type', 'Record support reactions at nodes.')
    at_rayleighForces = mka(MpcAttributeType.Boolean, 'rayleighForces', 'Response Type', 'Record Rayleigh forces at nodes.')

    # Optional Attributes Creation
    at_precision = mka(MpcAttributeType.Boolean, 'precision', 'Optional', 'Set the number of significant digits for output.')
    at_nSD = mka(MpcAttributeType.Integer, '$nSD', 'Optional', 'Number of significant digits for precision.')
    at_nSD.setDefault(10)
    at_time = mka(MpcAttributeType.Boolean, '-time', 'Optional', 'Include domain time in the first column of the output.')
    at_timeSeries = mka(MpcAttributeType.Boolean, 'timeSeries', 'Optional', 'Link the output to an existing TimeSeries tag.')
    at_tsTag = mka(MpcAttributeType.Index, '$tsTag', 'Optional', 'Tag of a previously constructed TimeSeries.')
    at_tsTag.indexSource.type = MpcAttributeIndexSourceType.Definition
    at_tsTag.indexSource.addAllowedNamespace("timeSeries")
    at_dT = mka(MpcAttributeType.Boolean, '-dT', 'Optional', 'Use dT to specify output interval for recording.')
    at_deltaT = mka(MpcAttributeType.Real, '$deltaT', 'Optional', 'Time step interval to control recording frequency.')
    at_deltaT.setDefault(1.0)
    at_CloseOnWrite = mka(MpcAttributeType.Boolean, '-CloseOnWrite', 'Optional', 'Close and reopen the file at each time step for live monitoring.')

    # Create the MetaData Object
    # This will create the GUI window
    xom = MpcXObjectMetaData()
    xom.name = 'ladruñoRecorder'

    # Add File Attributes
    xom.addAttribute(at_file_name)
    xom.addAttribute(at_file)
    xom.addAttribute(at_extension)

    # Add DOF Attributes
    xom.addAttribute(at_dof_X)
    xom.addAttribute(at_dof_Y)
    xom.addAttribute(at_dof_Z)

    # Add Node Attributes
    xom.addAttribute(at_nodes)

    # Add Response Type Attributes
    xom.addAttribute(at_disp)
    xom.addAttribute(at_vel)
    xom.addAttribute(at_accel)
    xom.addAttribute(at_incrDisp)
    xom.addAttribute(at_reaction)
    xom.addAttribute(at_rayleighForces)

    # Add Optional Attributes
    xom.addAttribute(at_precision)
    xom.addAttribute(at_nSD)
    xom.setVisibilityDependency(at_precision, at_nSD)
    xom.addAttribute(at_time)
    xom.addAttribute(at_timeSeries)
    xom.addAttribute(at_tsTag)
    xom.setVisibilityDependency(at_timeSeries, at_tsTag)
    xom.addAttribute(at_dT)
    xom.addAttribute(at_deltaT)
    xom.setVisibilityDependency(at_dT, at_deltaT)
    xom.addAttribute(at_CloseOnWrite)

    return xom

# ========================================
# HELPER FUNCTIONS
# ========================================



def extract_tags(mesh_domain, node_tags):
    """
    Extract unique and sorted node IDs from the mesh domain and update the provided list of node tags.
    
    Parameters:
    - mesh_domain: Object containing a collection of elements, where each element has nodes.
                   Assumes 'mesh_domain.elements' is iterable and 'element.nodes' contains node objects with an 'id' attribute.
    - node_tags (list): List to store unique node IDs. It is updated in-place.
    """
    # Use a set for faster lookup and ensure uniqueness
    unique_ids = set(node_tags)
    
    for element in mesh_domain.elements:
        for node in element.nodes:
            unique_ids.add(node.id)
    
    # Sort and update the original node_tags list
    node_tags[:] = sorted(unique_ids)


    """
    _summary_

    pinfo is a special object created and managed by STKO's scripting environment during the Tcl script generation process. It stands for "process info" and carries all the contextual information needed for script generation.
    The pinfo object is not imported; it is created and passed dynamically by STKO's backend when exporting Tcl scripts.

    Returns:
        _type_: _description_
    """

def writeTcl(pinfo):
    
	xobj = pinfo.analysis_step.XObject
    
	doc = PyMpc.App.caeDocument()
	if(doc is None):
		raise Exception('null cae document')

	ClassName = xobj.name
	if pinfo.currentDescription != ClassName:
		pinfo.out_file.write('\n{}# {} {}\n'.format(pinfo.indent, xobj.Xnamespace, ClassName))
		pinfo.currentDescription = ClassName

    # File configuration
    
	# File name
	file_name_at = xobj.getAttribute('$fileName')
	if(file_name_at is None):
		raise Exception('Error: cannot find "file" attribute')
	file_name = file_name_at.string

	# Extension definition
	at_extension = xobj.getAttribute('Extension')
	if(at_extension is None):
		raise Exception('Error: cannot find "Extension" attribute')
	extension = at_extension.string

	if '.' not in extension:
		raise Exception('Extension not valid')

	# File type definition
	file_type = ''
	at_file_type = xobj.getAttribute('-file')

	if(at_file_type is None):
		raise Exception('Error: cannot find "-file" attribute')
	file_type += '-file' if at_file_type.boolean else ''
	only_type = at_file_type.boolean



	#-----------------------------------------------------------|
	#---------------------------NODES---------------------------|
	#-----------------------------------------------------------|

	# Get nodes
	nodes_at = xobj.getAttribute('-node')
	if(nodes_at is None):
		raise Exception('Error: cannot find "nodes" attribute')
	SelectionSets = nodes_at.indexVector

	# Get node tags
	nodes_tags = []
	ele_tags = []
	for selection_set_id in SelectionSets:
		if not selection_set_id in doc.selectionSets:
			continue
		selection_set = doc.selectionSets[selection_set_id]

		for geometry_id, geometry_subset in selection_set.geometries.items():
			mesh_of_geom = doc.mesh.meshedGeometries[geometry_id]

			for domain_id in geometry_subset.vertices:
				domain = mesh_of_geom.vertices[domain_id]
				if domain.id not in nodes_tags:
					nodes_tags.append(domain.id)
			nodes_tags.sort()
			for domain_id in geometry_subset.edges:
				domain = mesh_of_geom.edges[domain_id]
				extract_tags(pinfo, domain, nodes_tags, xobj)


			for domain_id in geometry_subset.faces:
				domain = mesh_of_geom.faces[domain_id]
				extract_tags(pinfo, domain, ele_tags, xobj)


		for interaction_id in selection_set.interactions:
			domain = doc.mesh.meshedInteractions[interaction_id]
			extract_tags(pinfo, domain, nodes_tags, xobj)

	# Defining responses types
	respType = ''
	def_respTypes = [' disp',' vel',' accel',' incrDisp','eigen',' reaction',' rayleighForces']

	for resp_str in def_respTypes:
		if resp_str == 'eigen':
			eigen_at = xobj.getAttribute('eigen')
			mode_at = xobj.getAttribute('$mode')
			if eigen_at is None:
				raise Exception('Error: cannot find "{}" attribute'.format(resp_str))
			respType += ' eigen {}'.format(mode_at.integer) if eigen_at.boolean else ''
		else:
			response = xobj.getAttribute(resp_str[1:])
			if response is None:
				raise Exception('Error: cannot find "{}" attribute'.format(resp_str))
			respType += resp_str if response.boolean else ''

	if respType == '':
		raise Exception('Response Type cannot be empty')

	# Defining DOFs
	dofs = ''
	def_dofs = {'-dof_X':' 1','-dof_Y':' 2','-dof_Z':' 3'}
	for dof_str in def_dofs:
		dof_at = xobj.getAttribute(dof_str)
		if dof_at is None:
			raise Exception('Error: cannot find "{}" attribute'.format(dof_str))
		dofs += def_dofs[dof_str] if dof_at.boolean else ''
	if dofs == '':
		raise Exception('DOF cannot be empty')

	# Optionals 1
	sopt = ''
	at_time = xobj.getAttribute('-time')
	if(at_time is None):
		raise Exception('Error: cannot find "-time" attribute')
	sopt += ' -time ' if at_time.boolean else ''

	# Optionals 2
	at_precision = xobj.getAttribute('precision')
	at_nSD = xobj.getAttribute('$nSD')
	if(at_precision is None):
		raise Exception('Error: cannot find "precision" attribute')
	sopt += ' -precision {}'.format(at_nSD.integer) if at_precision.boolean else ''

	#Write TCL
	str_tcl = ''
	node_str = ''
	nodes_number = len(doc.mesh.nodes)
	ele_number = len(doc.mesh.elements)
	partitions = pinfo.process_count





	#-----------------------------------------------------------|
	#------------------WRITE RECORDER---------------------------|
	#-----------------------------------------------------------|
	# Create 'Results' folders
	path = os.path.abspath(pinfo.out_dir)

	# Create folder for Partitions Info files and check model
	os.makedirs(f'{path}/PartitionsInfo/info', exist_ok=True)
	with open(f'{path}\\PartitionsInfo\\info\\model_info.csv', 'w') as info_file:
		info_file.write(f'Number of nodes = {nodes_number}\n')
		info_file.write(f'Number of elements = {ele_number}\n')
		info_file.write(f'Number of partitions = {partitions}\n')

	# Parallel computing
	if pinfo.process_count > 1:
		process_block_count = 0
		for process_id in range(pinfo.process_count):
			first_done = False
			#Check if directories exist and create them if not
			os.makedirs(f'{path}/PartitionsInfo/coords', exist_ok=True)
			os.makedirs(f'{path}/PartitionsInfo/{respType[1:]}', exist_ok=True)

			#Add coordinate info
			if respType == ' disp':
				os.makedirs(f'{path}/Displacements', exist_ok=True)
				with open(f'{path}/PartitionsInfo/coords/coords_{process_id}.csv','w') as coords_file:
					coords_file.write('Node ID, X, Y, Z \n')
					for node_id in nodes_tags:
						if doc.mesh.partitionData.nodePartition(node_id) != process_id:
							continue
						nodex = doc.mesh.nodes[node_id].x
						nodey = doc.mesh.nodes[node_id].y
						nodez = doc.mesh.nodes[node_id].z
						coords_file.write("{} {} {} {} \n".format(node_id, nodex,nodey,nodez))
			elif respType == ' accel': os.makedirs(f'{path}/Accelerations', exist_ok=True)
			elif respType == ' reaction': os.makedirs(f'{path}/Reactions', exist_ok=True)

			#Add info about accelerations, displacements and reactions
			with open(f'{path}/PartitionsInfo/{respType[1:]}/{respType[1:]}_nodes_part-{process_id}.csv','w') as results_file:
				for node_id in nodes_tags:
					#In short, this line of code ensures that only nodes belonging to the current process running in parallel are processed.
					if doc.mesh.partitionData.nodePartition(node_id) != process_id:
						continue
					results_file.write(f'{node_id}\n')

					#This lines are about getting the output
					if not first_done:
						if process_block_count == 0:
							str_tcl += '\n{}{}{}{}\n'.format(pinfo.indent, 'if {$STKO_VAR_process_id == ', process_id, '} {')
						else:
							str_tcl += '{}{}{}{}\n'.format(pinfo.indent, ' elseif {$STKO_VAR_process_id == ', process_id, '} {')
						first_done = True
					str_tcl += '{}{}recorder Node {} "{}-node_{}-part_$STKO_VAR_process_id{}" -node {}{} -dof{}{}\n'.format(pinfo.indent, pinfo.tabIndent,file_type,file_name, node_id,extension, node_id , sopt,dofs, respType)
					if first_done:
						process_block_count += 1
				if process_block_count > 0 and first_done:
					str_tcl += '{}{}'.format(pinfo.indent, '}')

 	#TODO: this is not well implement
	# Secuencial computing
	else:
		os.makedirs(f'{path}/coords', exist_ok=True)
		os.makedirs(f'{path}/{respType[1:]}', exist_ok=True)
		if respType == ' disp':
				with open(f'{path}/coords/coords.csv','w') as coords_file:
					coords_file.write('Node ID, X, Y, Z \n')
					for node_id in nodes_tags:
						nodex = doc.mesh.nodes[node_id].x
						nodey = doc.mesh.nodes[node_id].y
						nodez = doc.mesh.nodes[node_id].z
						coords_file.write("{} {} {} {} \n".format(node_id, nodex,nodey,nodez))

		with open(f'{path}/{respType[1:]}/{respType[1:]}_nodes.csv','w') as results_file:
			for node_id in nodes_tags:
				results_file.write(f'{node_id}\n')
				node_str += f' {node_id}'
				str_tcl += '{}recorder Node {} "{}{}" -node {}{} -dof{}{}'.format(pinfo.indent,file_type,file_name,extension, node_str , sopt,dofs, respType,'\n')
	str_tcl += '{}{}'.format(pinfo.indent,'\n')
	pinfo.out_file.write(str_tcl)



