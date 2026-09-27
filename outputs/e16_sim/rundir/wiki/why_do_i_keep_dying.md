# Why do I keep dying?

Source: https://nethackwiki.com/wiki/Why_do_I_keep_dying%3F
Retrieved: 2026-08-25

This page is an attempt to provide basic tips for survival and it specifically describes typical beginner misconceptions regarding NetHack. It is aimed at new players who feel like they can't get the hang of it and die early every game, so it will focus on the early game stage and deliberately ignore NetHack's abundant corner cases. Follow the links if you want a complete strategy overview.
It is assumed you have played (and died) a couple of dozen times and know how to open doors and do simple stuff like that. If you are still wondering what the funny @ means, then first have a look at the excellent Guidebook that comes with NetHack, play a few games to get the feel of it, read the Guidebook again—picking up the numerous hints you will have overlooked during the first reading—then come back here.


== The one good game ==
You sometimes hear that NetHack is near impossible to win, because of the amount of exceptionally good dice rolls that are required to ascend: it takes years to get just one good game, with an early wand of wishing and having everything else work out just right. That is a myth!
If your survival depends on luck then you are following the wrong strategy. True, a few unavoidable deaths remain, like falling into a poisoned spiked pit, or the proverbial Gnome With the Wand of Death, but these are threatening only early, before countermeasures have been acquired. Truly outstanding players (such as Marvin) manage to ascend 80% of their games, and they do not get more wands of wishing than the rest of us. A large part of learning to play NetHack is acquiring safe habits.


== Moving fast, typing slowly ==
Never explore while Burdened or worse. Speed is a major issue. When you are Stressed, your average opponent can hit twice for your every chance to move, and your HP will just melt away. (If you do need to haul around heavy stashes of equipment, stick to known territory and be ready to shed weight at the sight of danger.) In such occasions, a wand of speed monster or speed boots are a big advantage, allowing you to hit (or escape) faster.
Do not use the arrow keys, because they provide only orthogonal movement, requiring twice as many steps to reach a diagonal destination. If the vi-like (yuhjklbn) keys do not suit you, make a habit of using the number pad instead. Take advantage of diagonal movement keys.
At the same time, type slowly and deliberately. You cannot run away more quickly from monsters by typing more quickly! Additionally, never hold down a key for autorepeat. When you hold down a key, you often stumble into something and try to attack or pass it faster than you can release the key. Instead, make use of the G, g, capital YUHJKLBN, and numpad 5 commands to go in one direction until you discover something interesting. Also useful is _. 
If you are running low on health and there are no monsters in the vicinity, rest and wait for your health to come back. The . command will let you rest for a number of turns that you can specify. You will stop resting as soon as anything dangerous (or interesting) comes along. 
You may also find it useful to take notes on what you are doing.  Besides the information's usefulness, it encourages you to be thoughtful.
Beginners should also play purposefully: "just exploring", in the long run, is the same as "just running around until you get killed". If you find yourself doing this, look at your equipment and status to see what your immediate goals should be: "I need an effective projectile weapon", "I need to get my armor class down", "I need to identify this magic stuff I've collected", "I need fire/cold/magic resistance before I go much further" ... whatever's appropriate at the moment. Thinking this way will improve your game rapidly.


== Traps ==
On dungeon level 1, every deadly trap has an automatically generated corpse on it. This decreases by 25% with each Dlvl, meaning that Dlvl 5 and below will never have traps with an auto-corpse. However, while you're on the level a monster may stumble onto a trap and die, so if you see a mysterious corpse that you didn't kill, beware.
If you're caught in a bear trap, you can escape five times faster by trying to move diagonally.
Darts, arrows, and rocks on the ground. Beware, these may be traps.
If "you hear rumbling in the distance", there is a rolling boulder trap ready to do d20 damage to anything in its path.
Darts and spiked pits have a chance of being poisoned and can cause insta-death. Since it takes a while to gain intrinsic poison resistance or identify the necessary jewelry, an alchemy smock is a great find in the early game. (If this is a frequent problem for you, it may be worth trying playing a barbarian as your first role, since they get poison resistance at level 1.)


== Finding down ==
Every room has a way out. If none is apparent, search for it by hitting s on every step along the wall or at the dead end of a corridor—it may take you ten times or so before finding a secret door or passage, though. If all doors are locked and you have no other means to open them, kick them down—but this is noisy and will wake sleeping monsters. If you kick down the door of a shop, the owner will attack. You can avoid kicking down shop doors by simply making sure that the dust in front of the door does not say "closed for inventory".
Every level has a way down. (There are exceptions, but not in the early dungeon.) Look on the map for a large, empty area where an additional, undiscovered room might fit and search (as above) along adjacent walls or suspiciously shaped corridors. Or the stairs could be covered by an item—yes, NetHack does have mighty big fortune cookies! If a monster is sitting on top of the stairs, you will see them as soon as the monster is out of sight. If you're pretty sure you've explored the whole level and still don't see any stairs, use the #terrain command to view the map without monsters or objects. Pressing _> will also show the stairs since Nethack 3.6.0 even if there are objects on top of them.


== Monsters too tough? ==
As you go deeper in the dungeon and level up, monsters that generate become more powerful. Monsters that can generate are determined by the average of your experience level and your dungeon level; see Monsters (by difficulty) for details. Even if you keep your experience level relatively low, you can still encounter difficult monsters as you venture deeper into the dungeon. If you are in a particularly deep area or a dangerous level such as the Oracle level, it may be in your best interest not to linger. Come back when you're more experienced or better equipped.


== Drinking water ==
Don't. A beginner who starved a couple of times might get the idea that drinking was also necessary. Unfortunately the Guidebook's advice on this matter is misleading ("Although creatures can survive long periods of time without food, there is a physiological need for water"). However, your character can in fact survive the whole game without drinking anything.
There is no need to drink water. In fact, quaffing potions of water is a big waste of resources; you are much better off saving them to turn into holy water. Drinking from fountains is downright dangerous, since many bad effects can occur. True, if you are exceptionally fortunate you might get an early wish, or some other benefit, but the chances are tiny. Much more likely, you will get nasty hostile monsters—water nymphs, water demons, or swarms of water moccasins.


== Eating ==
Eating corpses feels like Russian roulette. A kobold will poison you. A jackal "tastes terrible" but seems OK. But if you save another one for later, it'll give you deadly food poisoning. If you decide that eating corpses off the floor is uncivilized anyway and vow to stick to proper "people food", you'll probably starve before finding any.
First, understand that there are two separate kinds of poisoning that you can get from food. The first is food poisoning ("FoodPois"), contracted from eating old ("tainted") corpses. This will always kill you, but avoiding it is simple: Eat your corpses fresh. 60 turns is the limit! Only lichens, lizards, and corpses kept in an ice box do not age. By the same logic, remember that zombies and other undead died long before you met them: they are walking food poisoning. If your pet kills an enemy out of sight and leaves a dwarf corpse, it was probably a dwarf zombie; otherwise your pet would have eaten it.
The second kind is "regular" poison, which is simply a property of some monster types (e.g. kobolds). Corpses your pet dog or cat will eat are safe (with very few exceptions). Eating a poisonous corpse will lower your stats and HP, though it won't kill you directly. Unlike food poisoning, you can become resistant to this type, and in fact you should as soon as possible. This will also protect against spiked pits, the  poisoned arrows of Uruk-hai, and other sudden deaths. Barbarians, Healers, and orc characters start out resistant, while Monks gain the resistance at level 3; all others are safe as soon as "you feel healthy". (Watch out for other "you feel" messages, too, and learn their sources and effects.)
Remember: NetHack is not real life. Though most of us would not eat a sewer rat or an uncooked jackal corpse, and would get ill if we did, it is not an issue for your character. Unless there is something intrinsically harmful about a particular corpse, such as the poisonous kobold or the instantly petrifying cockatrice, and unless it is more than 60 turns dead, it will probably be safe for consumption.
There is also "proper" food, of course: Eggs and tripe rations are best left for pets, and keep the rest for hard times. In the roguelike community, such food is called "permafood", because it never rots.
Finally, if you are already at weak levels of nutrition (or worse, fainting), you can also #pray. You must take care not to anger your god, but if you really want to, it is quite possible to survive on prayer alone.


== Praying ==
Most real-life religions encourage you to pray regularly. But the Guidebook states clearly that you pray to the gods for help. The NetHack gods will be perfectly happy never to hear from you. If they regard you as constantly whining, they might become angry and eventually decide to put you out of your misery and send someone worthier to fetch the Amulet of Yendor for them.
Used sparingly, prayer can get you out of tight spots. Before praying again, wait around 700–1400 turns, the longer the better. Sacrifice can shorten the time, and it may also get you gifts.


== Watching your pet ==
Watching what corpses your cats and dogs eat will help you figure out which are safe—horses will not eat meat, and they can help you figure out which vegetarian corpses are safe (see diet). But there is more that your pet can do for you:
Finding better equipment is vital (see "Leveling up" above), but you must not Wear, wield or Put on anything that might be cursed, and altars for ascertaining that are scarce. Fortunately, your pet can indicate whether an item is cursed: just drop it on the floor where you can see it and wait. Pets will step on cursed items only reluctantly, if at all. If a pet walks over or picks up an item without a message appearing, it is safe to try on. Early on, pet-test most of the armor you find, to get your AC as low as you can.
Furthermore, your pet can kill a nymph before she can rob you or kill a peaceful coaligned unicorn that you must not desecrate yourself. Eventually, your pet's natural aggression may get it killed by the Minetown watch captain (a large dog or cat will attack watchmen) or a shopkeeper (warhorses attack these), so keep it away from these powerful monsters or polymorph it.


== Identifying ==
Boldly reading, quaffing and zapping everything you find is the obvious method of identification—and it is ridiculously, suicidally dangerous, so do not do it! Here is a quick summary of the item classes:
Armor, weapons and amulets are actually quite safe to try on after curse-testing them with your pet. A few items will autocurse, but they are rare and not life-threatening. Make sure you are capable of paying for the item if you decide to try it on in a shop, though.
Rings are also safe if curse-tested, with three complications: conflict, polymorph, and teleportation. Never try them near a shop or if your pet is powerful enough to kill you in one hit. If you're really afraid of a polymorph, take off your torso armor. Remove the ring immediately on the next turn to keep the chance of anything going wrong to a minimum, unless you think you can handle being polymorphed or teleported around. Often, putting on a ring will tell you nothing and you will have to identify it by some other means anyway.
Wands are fun. Write Elbereth with your finger (to exercise Wisdom), then add to the engraving with a wand. Most will identify or at least give you hints. Six wands give no message at all, are not particularly powerful, and can be further identified with a few common items. Never, ever put wands that make engravings vanish in your bag of holding until you are absolutely sure the wand is a wand of teleportation or a wand of make invisible, and not a wand of cancellation—otherwise you will blow up your bag of holding and all of its contents.
Potions are trickier. Potions of water are clear. Potions of oil light up when applied. By dipping some junk darts or arrows you may discover the potion of polymorph and the potion of sickness. Dipping a unicorn horn will turn three other harmful potions into water. Finally, you have to rule out the potions of sleeping and paralysis. Monsters may throw them at you, and you can also quaff-test them with sleep resistance or a worn ring of free action. All other potions are safe to quaff, provided that they are not cursed.
Scrolls are candidates for price identification, a complex and wearisome process. Fortunately, some of the most useful early-game scrolls are cheap and easy to price-identify. Drop the scroll at a shop (but don't actually sell it), and multiply the shopkeeper's offer by 2 (or 3 if you look like a tourist) to get the base price. (Some shopkeepers will offer only 3⁄4 of the normal selling price for unidentified items, however.) The scroll of identify is the most easily recognizable, the most common, and the cheapest, with a base price of 20. The mostly useless scroll of light has a base price of 50, and the scroll of enchant weapon has a base price of 60. The scroll of remove curse and the scroll of enchant armor both have a base price of 80. Several other scrolls can be identified from other factors:

Monsters will only ever read scrolls of teleportation, create monster, and (rarely) earth, which have obvious effects.
The scroll of teleportation is the only one that will be generated in closets, so it is the only one you may see lying on the ground outside a room.
The two scrolls on the first level of Sokoban are scrolls of earth.
A scroll of scare monster will be placed under the Sokoban prize, and it can also be observed from its effect on peaceful monsters. If a pet or a shopkeeper "turns to flee" for no apparent reason, you're standing on one.
The other scrolls are too complicated or risky to try out, so just read a blessed scroll of identify when you've collected enough.


== See also ==
Standard strategy
Standard strategy (SLASH'EM)
Role difficulty
Game stages
Notetaking
Furey's NetHack Tips


== External links ==
Guidebook for NetHack 3.6.1
