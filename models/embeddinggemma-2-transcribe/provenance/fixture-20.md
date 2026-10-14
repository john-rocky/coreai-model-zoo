# Fixture 20: the transcripts

The 20 LibriSpeech test-clean utterances of the gate (the reader fixtures `test-clean-20`, CC BY 4.0), read from
their float32 reference soft tokens. "Reader output" is reader v1's own PyTorch path, copied from the reader's release
note (`RELEASE.md` of mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader). The Core AI fp16 reader on the Mac GPU
(`reader/mac-gpu-gate.json`) writes the same transcript on 19 of the 20; on `1188-133604-0000` it reads "tiara" where
the PyTorch path reads "tintoret".

| Utterance | Word errors | Reader output |
|---|---:|---|
| `1089-134686-0000` | 2 / 28 | he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat muttered pieces to be ladled out in thick peppered flour fatten sauce |
| `1089-134686-0002` | 3 / 18 | after early nightfall fall all the yellow lamps would light up here and there the squalid quarter of the brawls |
| `1188-133604-0000` | 2 / 19 | you will find me continually speaking of four men tion sublime turner and tintoret in almost the same terms |
| `1188-133604-0001` | 2 / 18 | the unite every quality and sometimes you will find me referring to them as colorists sometimes as curio |
| `121-121726-0000` | 2 / 17 | also a popular can drives whereby love making may be suspended but not stopped during the picnic season |
| `121-121726-0001` | 1 / 8 | hurrah the tiresome product of a tireless tongue |
| `1221-135766-0000` | 1 / 37 | how strange it seemed to the sad woman as she watched the growth and the beauty that became every day more brilliant and the intelligence that through its quivering sunshine over the tiny features of this child |
| `1221-135766-0001` | 2 / 48 | god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonored bottom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven |
| `1284-1180-0000` | 0 / 24 | he wore blue silk stockings blue knee pants with gold buckles a blue ruffled waist and a jacket of bright blue braided with gold |
| `1284-1180-0001` | 1 / 26 | his hat had a peaked crown at a flat brim and around the brim was a row of tiny golden bells that tinkled when he moved |
| `1320-122612-0000` | 1 / 41 | since the period of art tale the active spirit of the country has surrounded it with a belt of rich and thriving settlements though none but the hunter or the savage is ever known even now to penetrate its wild recesses |
| `1320-122612-0001` | 0 / 30 | the dews were suffered to exhale and the sun had dispersed the mists and was shedding a strong and clear light in the forest when the travelers resumed their journey |
| `1580-141083-0000` | 2 / 30 | i will endeavor in my statement to avoid such terms as would serve to limit the events to any particular place or give a clew as to the people concerned |
| `1580-141083-0001` | 0 / 34 | i had always known him to be restless in his manner but on this particular occasion he was in such a state of uncontrollable agitation that it was clear something very unusual had occurred |
| `1995-1826-0000` | 0 / 27 | in the debate between the senior societies her defence of the fifteenth amendment had been not only a notable bit of reasoning but delivered with real enthusiasm |
| `1995-1826-0001` | 1 / 27 | the south she had not thought of seriously and yet knowing of its delightful hospitality and mild climate she was not averse to charlestown or new orleans |
| `2094-142345-0001` | 0 / 21 | but the windows are patched with wooden panes and the door i think is like the gate it is never opened |
| `2094-142345-0002` | 2 / 42 | for it is a solid heavy handsome door and must once have been in the habit of shedding with a sonorous bang behind the liveried lackey who had just seen his master and mistress off the grounds in a carriage and pair |
| `2300-131720-0000` | 1 / 13 | the paris plant like that of the crystal palace was a temporary exhibit |
| `2300-131720-0001` | 3 / 33 | the london plat was last temporary but not permanent supplying before it was torn out no fewer than three thousand lamps in hotels churches stores and dwellings in the vicinity of holborn |
