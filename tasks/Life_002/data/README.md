# Structure identities and the published CRISPR example

The two PDB files supplied here contain different types of structures:

| File | Identity | Protein chains in the supplied file |
|---|---|---|
| `7xg4.pdb` | Pseudomonas aeruginosa type IV-A CRISPR-Cas complex | `A B C D E F G H L` (nine protein chains, 2,876 CA records) |
| `6n40.pdb` | MmpL3 membrane protein from Mycobacterium smegmatis | `A` (one protein chain, 726 CA records) |

[RCSB 7XG4](https://www.rcsb.org/structure/7XG4) and [RCSB 6N40](https://www.rcsb.org/structure/6N40) document their identities. The biological assembly recorded in the supplied 6N40 file is a monomer with an identity transform.

## Original Foldseek-Multimer example

The [Foldseek-Multimer paper](https://doi.org/10.1038/s41592-025-02593-7), under *Complex structure prediction*, describes a **separate predicted Sulfitobacter sp. JL08 query**, comprising five Csf2 copies and one Csf3 copy. 7XG4 is a reported structural hit for that query. The supplied 6N40 structure is not that predicted query.

The paper's authors released the predicted query in [Zenodo record 13121434](https://doi.org/10.5281/zenodo.13121434), licensed CC BY 4.0:

- Archive: [`CRISPR.tar`](https://zenodo.org/records/13121434/files/CRISPR.tar), 2,202,112 bytes.
- Published archive MD5: `4e4923271514a34844a55e61f3b03efb`.
- Archive member: `CRISPR/pHS1543_NZ_CP025815_Csf3_5Csf2_unrelaxed_rank_001_alphafold2_multimer_v3_model_5_seed_000.pdb`.
- The member contains six protein chains (`A-F`) and 1,843 CA records. Its SHA-256 is `eeaded78739ca4790241eb65a03725e8c6dc2b513ed6b8b3900fa4c41e24693a`.
- [Original analysis scripts and reproduction instructions](https://github.com/steineggerlab/foldseek-multimer-analysis/tree/5e5d2cb029e6a623983f6ceed8f180574303297e/CRISPR_benchmark).

This source identifies the published positive example. The two packaged PDB files retain their existing coordinates and identities; using the published example requires obtaining the separate predicted query and explicitly recording which pair was aligned.
