def remove_duplicate_labels(ax, loc='lower right'):
    handles, labels = ax.get_legend_handles_labels()
    use_labels = []
    use_handles = []
    for j, l in enumerate(labels):
        if l not in use_labels:
            use_labels.append(l)
            use_handles.append(handles[j])
    ul, uh = zip(*sorted(zip(use_labels, use_handles)))
    ax.legend(uh, ul, numpoints=1, loc=loc)
