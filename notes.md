tmux new-session -d -s translate_offer -c /root/image-translation \
  "source venv/bin/activate && python translate_offer.py"

to start translate_offer.py

tmux kill-session -t translation"

tmux attach -t translation"
tmux attach -t translate_offer"