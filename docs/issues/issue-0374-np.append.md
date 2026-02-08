# #374: np.append

- **URL:** https://github.com/SciSharp/NumSharp/issues/374
- **State:** OPEN
- **Author:** @solarflarefx
- **Created:** 2019-12-05T20:29:23Z
- **Updated:** 2019-12-07T17:49:30Z
- **Labels:** help wanted, missing feature/s

## Description

Is there an equivalent method to np.append?  If not, what is the best workaround?

## Comments

### Comment 1 by @Nucs (2019-12-07T17:49:22Z)

Maybe np.concatenate, np.vstack or np.stack will help you. They might require reshaping to emulate np.append.

