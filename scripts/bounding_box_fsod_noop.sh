for run in $(seq 1 100); do
	echo "Running run $run"
	python bounding_box_fsod_noop_tree.py --subset-path subset_52
done