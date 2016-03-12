#include "FH5.h"

/*
  Copyright (c) 2012 Graeme Kennedy. All rights reserved. 
  Not for commercial purposes.
*/

FH5File::FH5File( MPI_Comm _comm ){
  fp = NULL;
  comm = _comm;
  file_offset = 0;
  file_end = 0;
  file_for_writing = 0;

  rfp = NULL;
  current = root = tip = NULL;
  num_comp = 0;
  comp_names = NULL;
}

FH5File::~FH5File(){
  if (rfp){ fclose(rfp); }
  if (root){ deleteFH5FileInfo(); }

  if (comp_names){
    for ( int k = 0; k < num_comp; k++ ){
      if (comp_names[k]){
        delete [] comp_names[k];
      }
    }
    delete [] comp_names;
  }
  
}

/*
  Open a file and write the pre-header information.

  Write the zone map to the file. This consists of a map between the
  zone numbers and the variable names.

  input:
  file_name: the file that will be created
  num_components:  the total number of zones
  component_names: the names of the components

  returns:
  0 on successs, 1 if there is an error creating the file

  Test if the 
*/
int FH5File::createFile( const char * file_name,
                         char ** component_names, 
                         int num_components ){
  if (fp){ 
    int rank;
    MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "[%d] FH5: Error, cannot create new file\n",
            rank);
    return 1;
  }
  else {
    // Open the file and make sure it opened properly
    fp = NULL;
    int slen = strlen(file_name)+1;
    char * fname = new char[ slen ];
    strcpy(fname, file_name);

    // Open the file for writing
    MPI_File_open(comm, fname, MPI_MODE_WRONLY | MPI_MODE_CREATE, 
                  MPI_INFO_NULL, &fp); 
    delete [] fname;

    file_for_writing = 1;
    file_end = 0;
    if (fp){
      file_offset = 0;

      // Write the zone and variable names to the file
      char datarep[] = "native";
      MPI_File_set_view(fp, file_offset, MPI_CHAR, MPI_CHAR,
                        datarep, MPI_INFO_NULL);   

      int rank;
      MPI_Comm_rank(comm, &rank);

      size_t header_len = (1 + num_components)*sizeof(int);
      for ( int k = 0; k < num_components; k++ ){
        if (component_names[k]){
          header_len += (strlen(component_names[k])+1)*sizeof(char);
        }
        else {
          // Create a default zone name
          char zone_name[128];
          header_len += (strlen(zone_name)+1)*sizeof(char);
        }
      }

      if (rank == 0){
        char * header = new char[ header_len ];
        // Record the number of components
        memcpy(header, &num_components, sizeof(int));
        header_len = sizeof(int);

        for ( int k = 0; k < num_components; k++ ){
          int slen = 0;
          // Create a default zone name
          char zone_name[128];

          // Copy the size of the string
          if (component_names[k]){
            slen = strlen(component_names[k])+1;
          }
          else {
            slen = strlen(zone_name)+1; 
          }
          memcpy(&header[header_len], &slen, sizeof(int));
          header_len += sizeof(int);

          // Copy over the zone variable name - note that sizeof(char) = 1
          if (component_names[k]){
            memcpy(&header[header_len], component_names[k], slen*sizeof(char));
          }
          else {
            memcpy(&header[header_len], zone_name, slen*sizeof(char));
          }
          header_len += slen*sizeof(char);
        }

        MPI_File_write(fp, header, header_len, MPI_CHAR, MPI_STATUS_IGNORE);
        delete [] header;
     }

      file_offset = header_len;
      return 0;
    }
  }

  return 1;
}

/*
  Write the data to a file.

  The file consists of a series of headers followed by two dimensional
  data. Note that dim1 may be different on all procs, but that dim2
  must be the same on all procs. The zone/variable names are only
  significant on the root processors.

  The headers are organized as follows:

  Header information:
  -------------------
  
  data type (int), 
  dim 1 (int), 
  dim 2 (int), 
  size zone name (int)
  size of variable (int)
  zone_name (char)
  variable name (char)
  
  Data:
  -----
  data (double) or (int) 
  
  dim1*dim2*sizeof(double)/sizeof(int)
*/
int FH5File::writeZoneData( char * zone_name, 
                            enum FH5DataNames data_name, 
                            char * var_names,
                            void * data, int dim1, int dim2 ){
  // Check the file status to ensure that it's open
  if (fp && file_for_writing){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    int *dim = new int[ size+1 ];
    dim[0] = 0;
    MPI_Allgather(&dim1, 1, MPI_INT, &dim[1], 1, MPI_INT, comm);

    for ( int k = 0; k < size; k++ ){
      dim[k+1] += dim[k];
    }
    int total_dim = dim[size];

    // Calculate the size of the buffer to use
    size_t header_len = 5*sizeof(int) + strlen(zone_name) + strlen(var_names) + 2;

    // Write the zone and variable names to the file
    char datarep[] = "native";
    MPI_File_set_view(fp, file_offset, MPI_CHAR, MPI_CHAR,
                      datarep, MPI_INFO_NULL);

    // Write the header for this zone just on the root processor
    if (rank == 0){
      // Allocate the pre-header to use
      char * pre_header = new char[ header_len ];
      int pre_int[5];
      pre_int[0] = data_name;
      pre_int[1] = total_dim;
      pre_int[2] = dim2;
      pre_int[3] = strlen(zone_name)+1;
      pre_int[4] = strlen(var_names)+1;
      
      // Copy the pre-header to the file
      memcpy(pre_header, pre_int, 5*sizeof(int));

      size_t off = 5*sizeof(int);
      memcpy(&pre_header[off], zone_name, strlen(zone_name)+1);

      off += strlen(zone_name)+1;
      memcpy(&pre_header[off], var_names, strlen(var_names)+1);

      MPI_File_write(fp, pre_header, header_len, MPI_CHAR, MPI_STATUS_IGNORE);
      delete [] pre_header;
    }

    // Increment the global file-offset to match
    file_offset += header_len;

    // Prepare to read in the data
    MPI_Datatype dtype = MPI_DOUBLE;
    if (data_name == FH5_INT){
      dtype = MPI_INT;
    }

    MPI_File_set_view(fp, file_offset, dtype, dtype,
                      datarep, MPI_INFO_NULL);   
    MPI_File_write_at_all(fp, dim[rank]*dim2, data, dim1*dim2, dtype, 
                          MPI_STATUS_IGNORE);

    if (dtype == MPI_DOUBLE){
      file_offset += total_dim*dim2*sizeof(double);
    }
    else if (dtype == MPI_INT){
      file_offset += total_dim*dim2*sizeof(int);
    }

    // Free the dimension count
    delete [] dim;

    return 1;
  }

  return 0;
}

void FH5File::close(){
  if (fp){
    MPI_File_set_size(fp, file_offset);
    MPI_File_close(&fp);
    fp = NULL;
  }
}

/*
  Open a file for reading 
*/
int FH5File::openFile( const char * file_name ){
  int rank, size = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  if (size != 1){
    fprintf(stderr, "[%d] FH5File: Error, cannot read file \
with more than one processor\n", rank);
  }
  else if (fp){ 
    fprintf(stderr, "[%d] FH5File: Error, cannot open file\n",
            rank);
    return 0;
  }
  else {
    // Open the file and make sure it opened properly
    rfp = fopen(file_name, "rb");

    if (rfp){
      scanFH5File();
      return 1;
    }
  }

  return 0;
}

/*
  Get the number of components defined in this file
*/
int FH5File::getNumComponents(){ 
  return num_comp; 
}

/*
  Return the component name, if defined
*/
char * FH5File::getComponentName( int comp ){ 
  if (comp >= 0 && comp < num_comp){
    return comp_names[comp];
  }
  return NULL;
}

/*
  Scan the FH5 file and obtain:
  
  1. The zone names
  2. The offsets into the file for the beginning of all data entries
  3. The size of the data entries
*/
void FH5File::scanFH5File(){
  if (!rfp){
    fprintf(stderr, "FH5File: Cannot scan file, NULL file pointer\n");
    return;
  }

  if (root){
    deleteFH5FileInfo();
  }

  if (comp_names){
    for ( int k = 0; k < num_comp; k++ ){
      if (comp_names[k]){ delete [] comp_names[k]; }
    }
    delete [] comp_names;
  }

  // Determine the file size
  fseek(rfp, 0, SEEK_END);
  int file_size = ftell(rfp);
  rewind(rfp);
  int file_pos = 0;

  // Read the header information
  num_comp = 0;
  if (fread(&num_comp, sizeof(int), 1, rfp) != 1){
    fprintf(stderr, "FH5: Error reading header\n");
    return;
  }

  comp_names = new char*[ num_comp ];
  for ( int k = 0; (k < num_comp) && !feof(rfp); k++ ){
    int slen = 0;
    if (fread(&slen, sizeof(int), 1, rfp) != 1){
      fprintf(stderr, "FH5: Error reading header\n");
      return;
    }
    
    size_t len = slen;
    comp_names[k] = new char[ len ];
    if (fread(comp_names[k], sizeof(char), len, rfp) != len){
      fprintf(stderr, "FH5: Error reading header\n");
      return;
    }
  }
  file_pos = ftell(rfp);

  root = current = tip = NULL;

  while (file_pos+1 < file_size){
    // Set the position in the file
    fseek(rfp, file_pos, SEEK_SET);

    // Read the header information 
    int header[5] = {0, 0, 0, 0, 0};  
    if (fread(header, sizeof(int), 5, rfp) != 5){
      fprintf(stderr, "FH5: Error scanning header\n");
      return;
    }

    // If no root exists, create a new one, otherwise append
    // to the link list
    if (root == NULL){
      tip = new FH5FileInfo();
      current = root = tip;
    }
    else {
      tip->next = new FH5FileInfo();
      tip = tip->next;
    }

    // Read the group name
    size_t slen = header[3];
    tip->zone_name = new char[ slen ];
    if (fread(tip->zone_name, sizeof(char), slen, rfp) != slen){
      fprintf(stderr, "FH5: Error reading group name\n");
      return;
    }

    // Record the type of data - one of the FH5DataNames
    tip->dtype = header[0];
    tip->dim1 = header[1];
    tip->dim2 = header[2];

    // Read the variable names
    slen = header[4];
    tip->var_names = new char[ slen ];
    if (fread(tip->var_names, sizeof(char), slen, rfp) != slen){
      fprintf(stderr, "FH5: Error reading variable names\n");
      return;
    }

    // Record the file position
    file_pos = ftell(rfp);
    tip->data_offset = file_pos;
    if (tip->dtype == FH5_INT){
      file_pos += sizeof(int)*tip->dim1*tip->dim2;
    }
    else {
      file_pos += sizeof(double)*tip->dim1*tip->dim2;
    }
  }
}

/*
  Delete the file information
*/
void FH5File::deleteFH5FileInfo(){
  current = root;
  while (current){
    root = current;
    current = current->next;
    delete root;
  }

  current = tip = root = NULL;
}

/*
  Set the pointer to work from the first zone in the file
*/
void FH5File::firstZone(){
  current = root;
}

/*
  Set the zone pointer to the next zone in the list
*/
int FH5File::nextZone(){
  if (current->next != NULL){
    current = current->next;
    return 1;
  }
  
  return 0;
}

/*
  Just get the variable names - if any
*/
int FH5File::getZoneInfo( const char ** zone_name,
                          const char ** var_names,
                          int * dim1, int * dim2 ){
  if (!current){
    return 0;
  }

  *zone_name = current->zone_name;
  *var_names = current->var_names;
  *dim1 = current->dim1;
  *dim2 = current->dim2;

  return 1;
}

/*
  Read the current group of data - both header info and actual data
*/
int FH5File::getZoneData( const char ** zone_name,
                          const char ** var_names,
                          void ** data, int * dim1, int * dim2 ){
  // No pointer or no file
  if (!current || !rfp){
    fprintf(stderr, "FH5File: Error, no file opened yet\n");
    return 0;
  }

  // Variables not defined in the file
  if (!current->var_names){
    fprintf(stderr, "FH5: Error variables are not defined\n");
    return 0;
  }

  // Check if this zone data is non-NULL
  fseek(rfp, current->data_offset, SEEK_SET);

  int dtype = current->dtype;
  *zone_name = current->zone_name;
  *var_names = current->var_names;
  *dim1 = current->dim1;
  *dim2 = current->dim2;

  size_t len = current->dim1*current->dim2;
  if (dtype == FH5_INT){
    *data = new int[ len ];
    if (fread(*data, sizeof(int), len, rfp) != len){
      fprintf(stderr, "FH5: Error reading integer data\n");
      return 0;
    }
  }
  else {
    *data = new double[ len ];
    if (fread(*data, sizeof(double), len, rfp) != len){
      fprintf(stderr, "FH5: Error reading double data\n");
      return 0;
    }
  }

  return 1;
}