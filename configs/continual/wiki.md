Two NetHack strategy references are on disk, in this worktree, next to this
prompt file:

    /root/nld/e15-wiki/configs/continual/wiki/why_do_i_keep_dying.md   (~15k chars)
    /root/nld/e15-wiki/configs/continual/wiki/standard_strategy.md     (~13k chars)

Read them with your IPython kernel before you look at the traces. They are
community strategy pages from NetHackWiki, retrieved 2026-08-25 -- general game
knowledge, NOT observations from these rollouts.

Use them for two things:

  1. NAMING what you see. The traces show behaviour; the wiki gives you the
     vocabulary and the causal account for it. A death you would otherwise write
     up as "died on dlvl 8" may be a recognised early-game failure with a known
     countermeasure.
  2. NOTICING what is absent. The wiki describes habits good players have. If
     the traces show the player never doing one of them, that absence is
     evidence you could not have found by reading the traces alone, because
     nothing in a trace points at what did not happen.

Two hard constraints, because the failure mode here is obvious:

  * EVERY entry you write must still be grounded in what THESE rollouts did.
    Cite the seed and turn numbers, as always. A correct piece of wiki advice
    that these games give you no evidence for is worth less than a specific
    observation, because the store holds only 6 entries per kind and a generic
    tip crowds out a measured one.
  * Do not transcribe. The player sees at most 180 characters per entry. "Read
    the wiki" is not an entry; "Elbereth costs one turn per letter and monsters
    attack throughout -- seed 8 died mid-word at T182-T202" is.

If the wiki contradicts what the traces show, say so in your rationale and
trust the traces: the wiki describes human play with a full keyboard and no
call budget, and this player has a restricted tool surface.
