# #349: Scipy.Signal

- **URL:** https://github.com/SciSharp/NumSharp/issues/349
- **State:** OPEN
- **Author:** @natank1
- **Created:** 2019-09-21T07:53:55Z
- **Updated:** 2019-09-22T15:17:25Z
- **Labels:** missing feature/s

## Description

Hi

Is there a way to run spectogram with scipy.signal in Using this library?

## Comments

### Comment 1 by @Nucs (2019-09-21T09:38:28Z)

No, Unfortunately we do not have implementations of [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html).
I would suggest you to use [Numpy.NET](https://github.com/SciSharp/Numpy.NET) as it wraps numpy directly and provides all its features.

### Comment 2 by @natank1 (2019-09-21T09:48:29Z)

Including SciPy ? 

### Comment 3 by @Nucs (2019-09-21T10:43:05Z)

Scipy is the name of the company that developed numpy...
אנחנו תומכים בעיקר באלגברה לינארית שזה בערך השימוש הכי נפוץ בסיפריה והכרחי בכדי להפעיל  את הסיפריה טנסורפלוו
הספריה [נמפיינט ](https://github.com/SciSharp/Numpy.NET) תומכת בכל הפיצ'רים של נמפיי

### Comment 4 by @natank1 (2019-09-21T13:03:15Z)

Sorry again

When you say the Scipy exists you mean we have an API for it under
Numpy.Net (I can find such ),
or we can take the python code of scipy and arrange it in away that
Numpy.net will handle this?
Sorry again fro bothering

On Sat, Sep 21, 2019 at 12:38 PM Eli Belash <notifications@github.com>
wrote:

> No, Unfortunately we do not have implementations of scipy.signal
> <https://docs.scipy.org/doc/scipy/reference/signal.html>.
> I would suggest you to use Numpy.NET
> <https://github.com/SciSharp/Numpy.NET> as it wraps numpy directly and
> provides all its features.
>
> —
> You are receiving this because you authored the thread.
> Reply to this email directly, view it on GitHub
> <https://github.com/SciSharp/NumSharp/issues/349?email_source=notifications&email_token=AB7W2Z45IIFTCL3AFD7MNP3QKXTRPA5CNFSM4IY5LVWKYY3PNVWWK3TUL52HS4DFVREXG43VMVBW63LNMVXHJKTDN5WW2ZLOORPWSZGOD7IOK2I#issuecomment-533783913>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AB7W2Z7XTZVASK5A6MN4YKLQKXTRPANCNFSM4IY5LVWA>
> .
>


### Comment 5 by @Nucs (2019-09-21T14:06:35Z)

Scipy is the name of the company that published Numpy. When you say Scipy it makes no sense.
When I comment I add URLs, just click on my mentions in the comments above of numpy.net or scipy.signal.
Numpy.NET is a different library from NumSharp which uses Pythonnet to call numpy.
Numpy.NET implements ALL numpy's functions but might be sometimes slower because you transfer data from C# to python.

### Comment 6 by @natank1 (2019-09-21T14:36:13Z)

So function /class that is written in python sipy.signal , how should be called inumpy.net?

> Scipy is the name of the company that published Numpy. When you say Scipy it makes no sense.
> When I comment I add URLs, just click on my mentions of numpy.net.
> Numpy.NET is a different library from NumSharp which uses Pythonnet to call numpy.
> Numpy.NET implements ALL numpy's functions but might be sometimes slower because you transfer data from C# to python.
> 
> —
> You are receiving this because you authored the thread.
> Reply to this email directly, view it on GitHub, or mute the thread.


### Comment 7 by @Nucs (2019-09-21T15:57:10Z)

I have no clue, This is NumSharp repository.
Read Numpy.NET's readme.md.

### Comment 8 by @natank1 (2019-09-21T16:06:56Z)

Thanks!
