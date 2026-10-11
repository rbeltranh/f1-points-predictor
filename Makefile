PYTHON        := uv run
SCRIPT        := get_dataset.py
RATE_LIMITED  := 75      # must match RATE_LIMITED in get_dataset.py
SLEEP_SECONDS := 1200    # 20 minutes
.PHONY: create-dataset create-dataset-bg dataset-log

# Run in the foreground, retrying after each rate limit until everything is downloaded
create-dataset:
	@while true; do \
		cd src/feature_processing; $(PYTHON) $(SCRIPT); \
		code=$$?; \
		if [ $$code -eq 0 ]; then \
			echo "All done."; break; \
		elif [ $$code -eq $(RATE_LIMITED) ]; then \
			echo "$$(date): rate limited, sleeping $(SLEEP_SECONDS)s..."; \
			sleep $(SLEEP_SECONDS); \
		else \
			echo "Script failed with exit code $$code, stopping."; exit $$code; \
		fi; \
	done

# Same thing in the background, immune to terminal closing and Mac idle sleep
create-dataset-bg:
	@caffeinate -i nohup $(MAKE) create-dataset > run.log 2>&1 &
	@echo "Started in background. Follow progress with: make dataset-log"

dataset-log:
	@tail -f run.log