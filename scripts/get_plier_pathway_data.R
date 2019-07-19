# Script to download preprocessed pathway data from PLIER
# repo and save it in a format that can be accessed in Python.
base_url <- 'https://github.com/wgmao/PLIER/blob'
commit <- '5351171779d32194dcacedb8e0e804021ea75fad'
output_dir <- file.path(getwd(), 'data', 'pathway_data')

canonical_file <- sprintf('%s/%s/data/canonicalPathways.rda?raw=true',
                          base_url, commit)
save_filename = file.path(output_dir, 'canonicalPathways.rda')
download.file(canonical_file, save_filename)
load(save_filename)
output_filename = file.path(output_dir, 'canonical_pathways.tsv')
write.table(canonicalPathways, file=output_filename,
            quote=FALSE, sep='\t')

oncogenic_file <- sprintf('%s/%s/data/oncogenicPathways.rda?raw=true',
                          base_url, commit)
save_filename = file.path(output_dir, 'oncogenicPathways.rda')
download.file(oncogenic_file, save_filename)
load(save_filename)
output_filename = file.path(output_dir, 'oncogenic_pathways.tsv')
write.table(oncogenicPathways, file=output_filename,
            quote=FALSE, sep='\t')
