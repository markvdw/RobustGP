infile=$1
cat $infile \
| jupytext --from py:percent --to ipynb --set-kernel - \
| papermill ${@:2} \
| jupyter nbconvert --TagRemovePreprocessor.enabled=True \
--TagRemovePreprocessor.remove_cell_tags="['remove_cell']" \
--TagRemovePreprocessor.remove_all_outputs_tags="['remove_output']" \
--stdin --no-input --output ${infile%.*}.html

# --TemplateExporter.exclude_code_cell=True \
