#load packages====
library(devtools)
library(ICTconverter)

#load functions====
convert_ict_to_df <- function(input_file, missing_values = c(-9999, -999999, -7777, -8888)) {
  # --- Step 1: Read first line (to find header line number) ---
  first_line <- readLines(input_file, n = 1)
  header_info <- suppressWarnings(as.numeric(strsplit(first_line, ",")[[1]]))
  header_line <- header_info[1]
  
  if (is.na(header_line)) stop(paste("Could not determine header line number for:", input_file))
  
  # --- Step 2: Extract column names from the header line ---
  all_lines <- readLines(input_file)
  column_line <- all_lines[header_line]
  column_names <- trimws(strsplit(column_line, ",")[[1]])
  
  # --- Step 3: Read the data starting one line after the header ---
  data <- read.table(
    input_file,
    sep = ",",
    skip = header_line,
    header = FALSE,
    fill = TRUE,
    comment.char = "",
    strip.white = TRUE,
    stringsAsFactors = FALSE
  )
  
  # --- Step 4: Assign column names ---
  colnames(data) <- column_names[1:ncol(data)]
  
  # --- Step 5: Replace missing values with NA ---
  for (mv in missing_values) {
    data[data == mv] <- NA
  }
  
  # --- Optional: Add filename as a column to track which file each row came from ---
  data$source_file <- basename(input_file)
  
  return(data)
}

combine_ict_files <- function(input_folder, output_file) {
  # Find all .ict files
  file_paths <- list.files(path = input_folder, pattern = "\\.ict$", full.names = TRUE)
  
  if (length(file_paths) == 0) {
    stop("No .ict files found in the specified folder.")
  }
  
  # Convert and combine all data frames
  all_data <- lapply(file_paths, convert_ict_to_df)
  
  # --- Step 6: Combine all datasets (align columns automatically) ---
  # combined_data <- do.call(rbind, lapply(all_data, function(df) {
  #   # Ensure consistent columns
  #   df <- df[, unique(unlist(lapply(all_data, names))), drop = FALSE]
  #   df
  # }))
  # 
  # # --- Step 7: Write one combined CSV ---
  # write.csv(combined_data, output_file, row.names = FALSE, na = "")
  # message("Combined ", length(file_paths), " files into one CSV: ", output_file)
  # invisible(combined_data)
  return(all_data)
}

#identify column names====
#get list of all converted ict dataframes
all_data <- combine_ict_files(
  input_folder = "C:\\Users\\megon\\OneDrive\\Documents\\Loyola\\4th Spring\\STAT 370\\Test",
  output_file = "C:\\Users\\megon\\OneDrive\\Documents\\Loyola\\4th Spring\\STAT 370\\Test\\Test.csv"
)

#extract column names from each dataframe
columns <- lapply(all_data, colnames)

#create dataframe with all column names
cols <- data.frame(matrix(nrow = 167, ncol = 0))
cols$names <- columns
cols$names <- sapply(cols$names, paste, collapse = ", ")