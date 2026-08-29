# Research validation and reproducibility audit

Generated from the repository contents on the audit date. Historical artifacts are preserved; no image, checkpoint, result, or metric was deleted or overwritten.

## Executive status

- Dataset images inspected: **3264**; unreadable: **0**.
- Exact duplicate groups: **166**; cross-legacy-split groups: **64**.
- Patient-level separation: **not verifiable**; no patient/subject/study/acquisition metadata was found in the supplied dataset index or paths.
- Historical training protocol: **test split used for model selection each epoch** in `src/brain_tumor_fusion/training/engine.py`.
- Clean protocol: **implemented but not run**; it assigns exact duplicate groups to train/validation/test and selects checkpoints on validation only.
- Runtime: **fresh compatible environment not yet available** on this machine; see the final report for exact blockers.

## 1. Data leakage and patient metadata

The 64 groups below are cryptographic exact matches: every listed file has the same SHA-256 digest, and the files were byte-compared within each group. This establishes identical encoded image files, not merely similar pixels. It does not establish that files with different bytes are different patients or acquisitions.

Class counts by legacy source split: `{'test': {'glioma_tumor': 100, 'meningioma_tumor': 115, 'no_tumor': 105, 'pituitary_tumor': 74}, 'train': {'glioma_tumor': 826, 'meningioma_tumor': 822, 'no_tumor': 395, 'pituitary_tumor': 827}}`.

No patient IDs, subject IDs, study IDs, acquisition IDs, or machine-readable group metadata were found. Filenames are generic image names and `data/captions.csv` contains only `image` and `caption` columns. Therefore, the new split prevents exact duplicate leakage only; patient-level leakage cannot be guaranteed.

Embedded-image metadata scan: **17** images contain EXIF fields (`{'ExifOffset': 17, 'ProcessingSoftware': 5, 'Software': 17, 'Orientation': 17, 'DateTime': 5, '59932': 5}`). The observed fields include software/orientation/timestamps and a custom tag; no direct patient identifier was observed in the values inspected, but EXIF must be reviewed and stripped or explicitly cleared by the data owner before release.

### Cross-split exact duplicate groups

#### Group 01

- SHA-256: `039d0eb6b8493db695929536aa7476f8c472381a7c20de981bfe7e0bdd875a82`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(100).jpg` | `test` | `no_tumor` | 13578 |
| `train/no_tumor/image(266).jpg` | `train` | `no_tumor` | 13578 |

#### Group 02

- SHA-256: `053a6036a57e58922ce011f713ddb44f78af390306aa055803e24338eb977075`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(28).jpg` | `test` | `no_tumor` | 15244 |
| `train/no_tumor/image(4).jpg` | `train` | `no_tumor` | 15244 |

#### Group 03

- SHA-256: `06e6f334b7e990bd685265cea5af2b11410235f4a3baad476c964249310330f3`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(35).jpg` | `test` | `no_tumor` | 11432 |
| `train/no_tumor/image(10).jpg` | `train` | `no_tumor` | 11432 |

#### Group 04

- SHA-256: `08c04f3659f54e170fe84f98a8ed10a235809251d6ba7c5372f08c90421ce51e`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(36).jpg` | `test` | `no_tumor` | 16191 |
| `train/no_tumor/image(11).jpg` | `train` | `no_tumor` | 16191 |

#### Group 05

- SHA-256: `136e27954eca1b7b887f3700f412b2c531153938a0d721d1557a94c3362497c2`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(15).jpg` | `test` | `no_tumor` | 13218 |
| `test/no_tumor/image(49).jpg` | `test` | `no_tumor` | 13218 |
| `test/no_tumor/image(53).jpg` | `test` | `no_tumor` | 13218 |
| `test/no_tumor/image(67).jpg` | `test` | `no_tumor` | 13218 |
| `test/no_tumor/image(84).jpg` | `test` | `no_tumor` | 13218 |
| `train/no_tumor/image(24).jpg` | `train` | `no_tumor` | 13218 |
| `train/no_tumor/image(28).jpg` | `train` | `no_tumor` | 13218 |
| `train/no_tumor/image(42).jpg` | `train` | `no_tumor` | 13218 |
| `train/no_tumor/image(59).jpg` | `train` | `no_tumor` | 13218 |

#### Group 06

- SHA-256: `16bdc3ca286cca1bb4761deeca41c63f5626ac65aff85231686caf5c3897c210`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(65).jpg` | `test` | `no_tumor` | 15016 |
| `train/no_tumor/image(40).jpg` | `train` | `no_tumor` | 15016 |

#### Group 07

- SHA-256: `18cfaf6cee35ef130675d36429a4d0d9d944cef9942c203b01975b7130ff6df5`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(63).jpg` | `test` | `no_tumor` | 12561 |
| `train/no_tumor/image(38).jpg` | `train` | `no_tumor` | 12561 |

#### Group 08

- SHA-256: `19d1e5fece32f25cffe40d78ee005f41ff3c2598c7765ed122ffb29518274326`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(37).jpg` | `test` | `no_tumor` | 9995 |
| `test/no_tumor/image(79).jpg` | `test` | `no_tumor` | 9995 |
| `train/no_tumor/image(12).jpg` | `train` | `no_tumor` | 9995 |
| `train/no_tumor/image(54).jpg` | `train` | `no_tumor` | 9995 |

#### Group 09

- SHA-256: `1d489fd63322dff8146dfb442c512362d2d42d5f4e44b90414202ac0abe5f97e`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(57).jpg` | `test` | `no_tumor` | 16334 |
| `train/no_tumor/image(32).jpg` | `train` | `no_tumor` | 16334 |

#### Group 10

- SHA-256: `1fe1db4adc37442001e9e054e3a990efa31b7db7cdb0a3d2711afbf346f9094b`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(27).jpg` | `test` | `no_tumor` | 14271 |
| `train/no_tumor/image(3).jpg` | `train` | `no_tumor` | 14271 |

#### Group 11

- SHA-256: `220816d55783431f776d10ccd11ebda7aa823602de035f0f07d0d7cd780cfa0f`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(39).jpg` | `test` | `no_tumor` | 13087 |
| `train/no_tumor/image(14).jpg` | `train` | `no_tumor` | 13087 |

#### Group 12

- SHA-256: `247420d663a2af95266ddfb5c9cd2ee741d2474f300ca0d25809d37f61e9c4df`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(58).jpg` | `test` | `no_tumor` | 13488 |
| `train/no_tumor/image(33).jpg` | `train` | `no_tumor` | 13488 |

#### Group 13

- SHA-256: `2c0036409ce103a8668735db595e18d01fbd821ca66e8ef43c77d6a4add544de`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(87).jpg` | `test` | `no_tumor` | 18772 |
| `train/no_tumor/image(62).jpg` | `train` | `no_tumor` | 18772 |

#### Group 14

- SHA-256: `2c6470345e908770e6f61b6a2e55cc80288f8598149e3b4558a1d79dfdba6c4a`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(47).jpg` | `test` | `no_tumor` | 12550 |
| `train/no_tumor/image(22).jpg` | `train` | `no_tumor` | 12550 |

#### Group 15

- SHA-256: `41d847d03ff8e8a80fde982029492adf6c622dc7d78fad2db48b14287c552296`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(11).jpg` | `test` | `no_tumor` | 7577 |
| `test/no_tumor/image(85).jpg` | `test` | `no_tumor` | 7577 |
| `train/no_tumor/image(60).jpg` | `train` | `no_tumor` | 7577 |

#### Group 16

- SHA-256: `49d44a7fd256a35903b981a4c7fbd2c3adc0f87f24ed0cf34d83a91aa13441b0`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(104).jpg` | `test` | `no_tumor` | 16158 |
| `train/no_tumor/image(272).jpg` | `train` | `no_tumor` | 16158 |

#### Group 17

- SHA-256: `4c6238dc9771a99b1623932047fd64d5eb26d25cc294916c516b85e261c8ab46`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(77).jpg` | `test` | `no_tumor` | 14041 |
| `train/no_tumor/image(52).jpg` | `train` | `no_tumor` | 14041 |

#### Group 18

- SHA-256: `4f2893125ab87fc6fcd6cd5388db893efeb9a1a30e59f942e7743144766c78f6`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(34).jpg` | `test` | `no_tumor` | 13753 |
| `train/no_tumor/image(9).jpg` | `train` | `no_tumor` | 13753 |

#### Group 19

- SHA-256: `4f3c00ef520660295576c472a0cafc9f1bb400d736b352b736966af08bf09235`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(21).jpg` | `test` | `no_tumor` | 10600 |
| `test/no_tumor/image(69).jpg` | `test` | `no_tumor` | 10600 |
| `train/no_tumor/image(44).jpg` | `train` | `no_tumor` | 10600 |

#### Group 20

- SHA-256: `53765f8fc313f7684b0df4547a89e27ecae20ae66c60ee26318fc9f1a301de8d`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(97).jpg` | `test` | `no_tumor` | 23703 |
| `train/no_tumor/image(262).jpg` | `train` | `no_tumor` | 23703 |

#### Group 21

- SHA-256: `5b32b70fee5498a7e242f40fd975486691a248ef53c7657ac4c5831133ade0fb`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(78).jpg` | `test` | `no_tumor` | 19344 |
| `train/no_tumor/image(171).jpg` | `train` | `no_tumor` | 19344 |
| `train/no_tumor/image(238).jpg` | `train` | `no_tumor` | 19344 |
| `train/no_tumor/image(292).jpg` | `train` | `no_tumor` | 19344 |
| `train/no_tumor/image(53).jpg` | `train` | `no_tumor` | 19344 |

#### Group 22

- SHA-256: `5dc458fca643f408eb0da30575e47f363f70ce78a3a14f1b8452139696512b00`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(38).jpg` | `test` | `no_tumor` | 18147 |
| `train/no_tumor/image(13).jpg` | `train` | `no_tumor` | 18147 |

#### Group 23

- SHA-256: `6d94cb16b0e532e10913e0f7e82bc9f0e2ddc3ff08601f8f7b118b88a137e111`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(91).jpg` | `test` | `no_tumor` | 10666 |
| `train/no_tumor/image(66).jpg` | `train` | `no_tumor` | 10666 |

#### Group 24

- SHA-256: `6df56cc8a677bbc089ccb7f85e5f40b665a551aa28c7f2fffffcaddf521ed503`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(42).jpg` | `test` | `no_tumor` | 10976 |
| `train/no_tumor/image(17).jpg` | `train` | `no_tumor` | 10976 |

#### Group 25

- SHA-256: `6e643c3263775b4637ad4e11620277c6ac65dcf63996826f76cac2dac5d8afdc`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(41).jpg` | `test` | `no_tumor` | 12598 |
| `test/no_tumor/image(52).jpg` | `test` | `no_tumor` | 12598 |
| `train/no_tumor/image(16).jpg` | `train` | `no_tumor` | 12598 |
| `train/no_tumor/image(27).jpg` | `train` | `no_tumor` | 12598 |

#### Group 26

- SHA-256: `733e77e4b6da2d851bddbf9cf2768c02db39d5fa0f7be61fdbcb9d1e044abb76`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(46).jpg` | `test` | `no_tumor` | 11184 |
| `test/no_tumor/image(50).jpg` | `test` | `no_tumor` | 11184 |
| `test/no_tumor/image(54).jpg` | `test` | `no_tumor` | 11184 |
| `train/no_tumor/image(21).jpg` | `train` | `no_tumor` | 11184 |
| `train/no_tumor/image(25).jpg` | `train` | `no_tumor` | 11184 |
| `train/no_tumor/image(29).jpg` | `train` | `no_tumor` | 11184 |

#### Group 27

- SHA-256: `76a70a3f68103e335f080a323d763fbda5febc37e4714c2f9b54499795f13fa3`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(32).jpg` | `test` | `no_tumor` | 16209 |
| `test/no_tumor/image(64).jpg` | `test` | `no_tumor` | 16209 |
| `train/no_tumor/image(39).jpg` | `train` | `no_tumor` | 16209 |
| `train/no_tumor/image(7).jpg` | `train` | `no_tumor` | 16209 |

#### Group 28

- SHA-256: `7703b9670ac9265886cbf62735e2ff074c1507a654573807d0a8468a54c89dd7`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(43).jpg` | `test` | `no_tumor` | 15584 |
| `train/no_tumor/image(18).jpg` | `train` | `no_tumor` | 15584 |

#### Group 29

- SHA-256: `77d1374baf9988fd4f9272886783a65820b3afbbbaa773777ed01a4b04ff9d80`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(88).jpg` | `test` | `no_tumor` | 11593 |
| `train/no_tumor/image(63).jpg` | `train` | `no_tumor` | 11593 |

#### Group 30

- SHA-256: `7b73d6019deb1482bc5fb7027f12719bea983c8eb0b0c0f3e95feb61c90fafb3`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(22).jpg` | `test` | `no_tumor` | 15347 |
| `test/no_tumor/image(61).jpg` | `test` | `no_tumor` | 15347 |
| `test/no_tumor/image(71).jpg` | `test` | `no_tumor` | 15347 |
| `train/no_tumor/image(36).jpg` | `train` | `no_tumor` | 15347 |
| `train/no_tumor/image(46).jpg` | `train` | `no_tumor` | 15347 |

#### Group 31

- SHA-256: `812a2c6599255067e9752c7d0037f409d91f7336f42d7af3440fb5c2c91e8da1`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(72).jpg` | `test` | `no_tumor` | 14039 |
| `train/no_tumor/image(47).jpg` | `train` | `no_tumor` | 14039 |

#### Group 32

- SHA-256: `9251febacd510bc6ab8fdebb28ac58c8db217df4d5e8af285a99547b2d3b8584`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(44).jpg` | `test` | `no_tumor` | 14533 |
| `train/no_tumor/image(19).jpg` | `train` | `no_tumor` | 14533 |

#### Group 33

- SHA-256: `94721a1b47a80cb0533acdbac52f92ad7614d4685284b74b2a96914b038947a9`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(31).jpg` | `test` | `no_tumor` | 7143 |
| `test/no_tumor/image(68).jpg` | `test` | `no_tumor` | 7143 |
| `train/no_tumor/image(43).jpg` | `train` | `no_tumor` | 7143 |

#### Group 34

- SHA-256: `9af05acd768793e89fc6776c3af838b0bf0747cfc3a3737e1cd3af44ea09c930`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(10).jpg` | `test` | `no_tumor` | 15159 |
| `test/no_tumor/image(33).jpg` | `test` | `no_tumor` | 15159 |
| `test/no_tumor/image(66).jpg` | `test` | `no_tumor` | 15159 |
| `test/no_tumor/image(81).jpg` | `test` | `no_tumor` | 15159 |
| `train/no_tumor/image(41).jpg` | `train` | `no_tumor` | 15159 |
| `train/no_tumor/image(56).jpg` | `train` | `no_tumor` | 15159 |
| `train/no_tumor/image(8).jpg` | `train` | `no_tumor` | 15159 |

#### Group 35

- SHA-256: `9c3b00e6b1f5fcb23d3ebf1a8b04f9517691cc7b7e2cfdbde375625c920b97a6`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(29).jpg` | `test` | `no_tumor` | 13004 |
| `train/no_tumor/image(5).jpg` | `train` | `no_tumor` | 13004 |

#### Group 36

- SHA-256: `9d338079dd3288665657816ebeec6ada298761e0ff9f4c22a59f05c46768db14`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(24).jpg` | `test` | `no_tumor` | 11990 |
| `test/no_tumor/image(70).jpg` | `test` | `no_tumor` | 11990 |
| `train/no_tumor/image(45).jpg` | `train` | `no_tumor` | 11990 |
| `train/no_tumor/image.jpg` | `train` | `no_tumor` | 11990 |

#### Group 37

- SHA-256: `a8155f8b8f640f8464b58a380026d01471800d15e76cbfd9678848d669c7e3a7`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(62).jpg` | `test` | `no_tumor` | 16244 |
| `train/no_tumor/image(37).jpg` | `train` | `no_tumor` | 16244 |

#### Group 38

- SHA-256: `a8d8bd68e37588a5be079ea92de410d9097988aea40552f6fd08ca44a7dd6c74`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(80).jpg` | `test` | `no_tumor` | 13121 |
| `test/no_tumor/image(9).jpg` | `test` | `no_tumor` | 13121 |
| `train/no_tumor/image(55).jpg` | `train` | `no_tumor` | 13121 |

#### Group 39

- SHA-256: `ad26946bf8b6ba5be922f24ab271337964fc35675fa4d0398719e33665337ece`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(92).jpg` | `test` | `no_tumor` | 9683 |
| `train/no_tumor/image(67).jpg` | `train` | `no_tumor` | 9683 |

#### Group 40

- SHA-256: `affc1b4d529e3c1f06052e6fe5d27ef659071f0c1b14b4197c6d2c4f30ae93bb`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(99).jpg` | `test` | `no_tumor` | 16654 |
| `train/no_tumor/image(325).jpg` | `train` | `no_tumor` | 16654 |

#### Group 41

- SHA-256: `b23fe6dad4ff5e7e0fe80d1d315399658298647951c5f5e148e8c0d6ff6e3b5e`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(95).jpg` | `test` | `no_tumor` | 11459 |
| `train/no_tumor/image(70).jpg` | `train` | `no_tumor` | 11459 |

#### Group 42

- SHA-256: `b577189346bc4cc74942150e5875bead25153aa474db97d1882ebf72244de3ec`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(94).jpg` | `test` | `no_tumor` | 14964 |
| `train/no_tumor/image(69).jpg` | `train` | `no_tumor` | 14964 |

#### Group 43

- SHA-256: `bf590df79c3f0a0ee28f4c1238720e89f40e4437c827273e3a168708f6fe8476`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(93).jpg` | `test` | `no_tumor` | 13291 |
| `train/no_tumor/image(68).jpg` | `train` | `no_tumor` | 13291 |

#### Group 44

- SHA-256: `c3f2d52e1a996bea716774c0051492202d3bdf76fdb53f9ade2b5c6289a6cb36`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(48).jpg` | `test` | `no_tumor` | 10709 |
| `test/no_tumor/image(75).jpg` | `test` | `no_tumor` | 10709 |
| `train/no_tumor/image(23).jpg` | `train` | `no_tumor` | 10709 |
| `train/no_tumor/image(50).jpg` | `train` | `no_tumor` | 10709 |

#### Group 45

- SHA-256: `c4c847d6e586faffcadf669ff0a9a608542741edc3a387040c059df1bb7052bb`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(56).jpg` | `test` | `no_tumor` | 12620 |
| `train/no_tumor/image(31).jpg` | `train` | `no_tumor` | 12620 |

#### Group 46

- SHA-256: `c50e621a36b08a9c2f28e239dc00b27ce321bd8ed715a48ced97515e965473d2`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(96).jpg` | `test` | `no_tumor` | 12988 |
| `train/no_tumor/image(71).jpg` | `train` | `no_tumor` | 12988 |

#### Group 47

- SHA-256: `c515996f4519dbd88019480d9e6442c71bc2513a7e2cb35a776b56af27b1021a`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(98).jpg` | `test` | `no_tumor` | 12914 |
| `train/no_tumor/image(264).jpg` | `train` | `no_tumor` | 12914 |

#### Group 48

- SHA-256: `c6b37dbcc8c59a18f68c4ecde7f081eb0caa8cae6edb6d438e11170f0736bf01`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(26).jpg` | `test` | `no_tumor` | 12991 |
| `train/no_tumor/image(2).jpg` | `train` | `no_tumor` | 12991 |

#### Group 49

- SHA-256: `c6d35e130bc719af5e19824979ce640e835635fc495a7383435df8b9aa16b023`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(74).jpg` | `test` | `no_tumor` | 12461 |
| `train/no_tumor/image(49).jpg` | `train` | `no_tumor` | 12461 |

#### Group 50

- SHA-256: `cae1e3ac8a29d55a9f7c0638c1d26be1d91a18c26715459dfcfa56a26317c11b`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(82).jpg` | `test` | `no_tumor` | 18286 |
| `train/no_tumor/image(57).jpg` | `train` | `no_tumor` | 18286 |

#### Group 51

- SHA-256: `d07ac7d4c62a45c3ca88e4b57cf485acd22b5d5f5e519110f769eb0cd9169d47`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(16).jpg` | `test` | `no_tumor` | 12060 |
| `test/no_tumor/image(45).jpg` | `test` | `no_tumor` | 12060 |
| `test/no_tumor/image(55).jpg` | `test` | `no_tumor` | 12060 |
| `test/no_tumor/image(60).jpg` | `test` | `no_tumor` | 12060 |
| `train/no_tumor/image(20).jpg` | `train` | `no_tumor` | 12060 |
| `train/no_tumor/image(30).jpg` | `train` | `no_tumor` | 12060 |
| `train/no_tumor/image(35).jpg` | `train` | `no_tumor` | 12060 |

#### Group 52

- SHA-256: `d08b052913eb262dfd31fdd6d7ca904c5343087cdc458ca928c9e85bf41bf388`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(86).jpg` | `test` | `no_tumor` | 15401 |
| `train/no_tumor/image(61).jpg` | `train` | `no_tumor` | 15401 |

#### Group 53

- SHA-256: `d8ea17fdb98908199591b41310c56ae6a2b90408287403a45333653e079944b1`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(103).jpg` | `test` | `no_tumor` | 20868 |
| `train/no_tumor/image(270).jpg` | `train` | `no_tumor` | 20868 |

#### Group 54

- SHA-256: `d9f0d5ef8784abb3ac7cadb3e31a9b1ef4879de8edb20a637d66c49f6c8eb60f`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(90).jpg` | `test` | `no_tumor` | 24249 |
| `train/no_tumor/image(65).jpg` | `train` | `no_tumor` | 24249 |

#### Group 55

- SHA-256: `df46c96890474f2ec5f08aa2c32e6bc1c6b5685f8b913501f954a515a6de02fa`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(40).jpg` | `test` | `no_tumor` | 26104 |
| `train/no_tumor/image(15).jpg` | `train` | `no_tumor` | 26104 |

#### Group 56

- SHA-256: `dfbba0fc3e25b2cc10d6fde8faf975c6f475baa770ede5ea3fd1e495995736ef`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(51).jpg` | `test` | `no_tumor` | 20420 |
| `train/no_tumor/image(26).jpg` | `train` | `no_tumor` | 20420 |

#### Group 57

- SHA-256: `e34e448b7509d16d43c6a6505995cdbb7bf6b2ad29ee09a37ec740c8494f74e3`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(59).jpg` | `test` | `no_tumor` | 15427 |
| `train/no_tumor/image(34).jpg` | `train` | `no_tumor` | 15427 |

#### Group 58

- SHA-256: `e834d68ccf89fc7d5f8a93f0718fb614dd3591102ee4ed5c5d851ac43dbd9e1c`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(102).jpg` | `test` | `no_tumor` | 17032 |
| `train/no_tumor/image(268).jpg` | `train` | `no_tumor` | 17032 |

#### Group 59

- SHA-256: `eed488f2e399ac4a011a52b70eff0a73837f1f2b8ef02297841792ba819420bc`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(76).jpg` | `test` | `no_tumor` | 14987 |
| `train/no_tumor/image(51).jpg` | `train` | `no_tumor` | 14987 |

#### Group 60

- SHA-256: `f0f59157eff868aa8dc74d5a6b5d384421c8587b5f15d90ca8414f4650276436`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(89).jpg` | `test` | `no_tumor` | 22444 |
| `train/no_tumor/image(64).jpg` | `train` | `no_tumor` | 22444 |

#### Group 61

- SHA-256: `f1aa73c33c56a13b8fe3b256ba2e6e3d2d84ddc6f3ed814d874bc88726ac1f9d`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(101).jpg` | `test` | `no_tumor` | 14906 |
| `train/no_tumor/image(326).jpg` | `train` | `no_tumor` | 14906 |

#### Group 62

- SHA-256: `f2d9f0979df98ce2e4aab669d3c141ca02ebce9eb15775267bfd6d4eae2a7d7f`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(25).jpg` | `test` | `no_tumor` | 14044 |
| `test/no_tumor/image(73).jpg` | `test` | `no_tumor` | 14044 |
| `train/no_tumor/image(1).jpg` | `train` | `no_tumor` | 14044 |
| `train/no_tumor/image(48).jpg` | `train` | `no_tumor` | 14044 |

#### Group 63

- SHA-256: `f3bbcbc551b2533f763091b8d11957d40118b57fe937c3b940f3f8b26098030f`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(30).jpg` | `test` | `no_tumor` | 14079 |
| `train/no_tumor/image(6).jpg` | `train` | `no_tumor` | 14079 |

#### Group 64

- SHA-256: `ff1df5849a35548de3ed553bc8c927511045cce13c42a2159c62f546c52230c9`
- Class folder(s): `no_tumor`; legacy membership: `test, train`
- Exact byte identity: **verified = true**
- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.

| File | Legacy split | Class | Bytes |
|---|---|---|---:|
| `test/no_tumor/image(83).jpg` | `test` | `no_tumor` | 16956 |
| `train/no_tumor/image(58).jpg` | `train` | `no_tumor` | 16956 |

## 2. Train/validation/test audit

The historical `try_engine_train` loop builds `train_loader` and `test_loader`, evaluates the latter after every training epoch, and saves the best checkpoint using that test accuracy. Consequently, the historical test set is used for model selection and repeated during experiments. The separate historical `scripts/evaluation/evaluate.py` also evaluates a selected checkpoint on that same test directory.

The clean path is `scripts/data/create_leakage_free_split.py` followed by `scripts/training/train_clean.py` with `configs/clean_resnet18_image_only.yaml`. It does not move or delete raw images. It uses all raw images as source records, keeps each SHA-256 group intact, and defines `train -> validation model selection -> final test once`. Since patient metadata is absent, it is an exact-duplicate-safe split, not a patient-level split.

The generated manifest contains **3264** records with exact-hash groups crossing new splits: **0**. New split counts are: `{'test/glioma_tumor': 139, 'test/meningioma_tumor': 141, 'test/no_tumor': 75, 'test/pituitary_tumor': 135, 'train/glioma_tumor': 648, 'train/meningioma_tumor': 655, 'train/no_tumor': 350, 'train/pituitary_tumor': 630, 'validation/glioma_tumor': 139, 'validation/meningioma_tumor': 141, 'validation/no_tumor': 75, 'validation/pituitary_tumor': 136}`.

## 3. Checkpoint verification

Inspected checkpoint artifacts: **69**. A machine-readable inventory with file sizes, SHA-256 digests, serialization type, and payload string signatures is in [`checkpoint_manifest.csv`](checkpoint_manifest.csv). Checkpoint files were not loaded because the only available compatible-looking package set is inside the invalid project `.venv`, which must not be used; system Python has no PyTorch.

| Experiment/config reference | Checkpoint reference | Metrics artifact | Verification status |
|---|---|---|---|
| `configs/resnet50_image_only.yaml` | `best_ep18_acc0.830.pt`, `final.pt` | `outputs/results/fusion_classification_report.json` and `test_predictions.csv` | **Unverifiable/conflicting**: the result artifact recomputes to 0.2766497462 accuracy, while the filename claims 0.830; no trusted log/config-to-checkpoint record ties them together. |
| `_archive/legacy_code/fine_tune.yaml` | `best_ep22_acc0.810.pt` | no independent matching result artifact found | **Unverifiable**: filename claim only. |
| `configs/resnet18_image_only.yaml` | no uniquely mapped checkpoint | none | **Unverifiable**: multiple checkpoint families exist without experiment manifests or logs. |
| `configs/clean_resnet18_image_only.yaml` | none yet | none | **Not run**: clean pipeline is code/config only at this stage. |

Filenames such as `best_ep18_acc0.830.pt` are therefore treated as labels, not evidence. No filename claim has been promoted to a scientific result.

## 4. Result consistency audit

The active historical prediction CSV contains 394 labeled predictions. Recomputing metrics from its `label_true`/`label_pred` columns gives:

- Accuracy: **0.2918781725888325** (115/394).
- Confusion matrix: `[[0, 100, 0, 0], [0, 115, 0, 0], [0, 105, 0, 0], [0, 74, 0, 0]]` in class order `['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']`.
- The active JSON report and archived JSON report contain the same 27.66497462% accuracy and matching per-class values; this is a historical artifact, not a newly validated final performance claim.

Archived report accuracy field: `0.2766497461928934`; it agrees with the recomputed historical CSV but does not resolve the checkpoint conflict.

Other repository findings: archived visualization scripts explicitly contain example/dummy or hard-coded metric data; the archived dataset-distribution graphic used a no-tumor count inconsistent with the inspected files; and the archived sample-prediction visualization generated random placeholder images/confidences. These remain archived and are not evidence.

## 5. Reproducibility and release limitations

- Dataset provenance, license, consent/de-identification statement, and acquisition protocol are absent from the supplied project and must be supplied by the dataset owner.
- Patient-level separation cannot be guaranteed without group metadata.
- Historical results used the test split for model selection and repeated evaluation.
- Text fusion is not implemented; the text encoder is a placeholder and captions are class-derived templates, so active claims are image-only.
- Checkpoint-to-config-to-metric lineage is incomplete; existing filenames are not independently verified.
- No clean pipeline training/evaluation run was performed in this audit, so no new performance number is reported.

## 6. Release gate

Before publication or GitHub release, obtain dataset provenance/permission, establish patient/subject grouping if available, regenerate the clean manifest, run the clean pipeline in a compatible fresh environment, save a config/seed/manifest/checkpoint/metrics manifest, and review every generated figure for provenance. Do not commit raw medical images, checkpoints, model weights, logs, or private metadata.
