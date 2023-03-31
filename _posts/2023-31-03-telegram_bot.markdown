---
layout: post
title: "Telegram Bot"
date: 2023-03-31 10:16
tags: programming
---

In this post I will describe how I built a telegram bot for a weekly coffee pairing of members in a telegram chat group.
The bot was built for a telegram community in London focused on helping Ukrainian professionals who moved to the UK due to the war, to assimilate in the UK and specifically find a job. I volunteered to do it despite having no past experience of building telegram bots. The bot was deployed after 4 days of on/off development and feedback from the community founder.

The bot has already generated 3 successful pairings, I hope people enjoyed their random chats and got something useful from them.
For me it was more of a fun challenge as well as wanting to help out.

# Basic idea

The idea behind the random coffee pairing is to allow members of the community, around 800 of them, to have a chat over coffee with other members of the community, allowing the community to be more connected at the individual level, beyond the general events that are organized every week. Finally, also to allow people to benefit from other members' experience, skills and knowledge specifically about job search but also life in general.

# The scope

As the community was founded as a private telegram channel where the members can communicate, the aim was to create a telegram bot with the following specification:

1. A user sends a /start message to the bot
2. The bot explains its purpose, and what are the available commands. This includes:

- **/join** - ask to join the random pairings
- **/remove** - ask to be removed from the random pairings
- **/pause** - ask to pause participation in the weekly pairings, but still remain in the database.
- **/resume** - resume the participation in the weekly pairings.

3. A user sends a /join command to join the random coffee pairings.
4. The bot prompts the user to answer a few questions about themselves including: name, occupation, social media and linkedin links. The user can skip some of the questions. The bot then saves the responses and the user in a database.
5. A scheduled job is triggered once a week at a defined time, the job consists of a function that reads all willing participants, reads all past parings, and creates random pairs of users. The function then asks the bot to send messages to the individuals in each pair informing them on their random pairings including their telegram @'s and general provided info.

# Tech stack

Telegram is quite a versatile messaging platform, with many features including chat groups/communities as well as bots.
A telegram bot is a virtual user that can be programmed to respond to certain commands by sending messages, accessing APIs etc. The creation of a telegram bot is very simple, Telegram has a very good documentation for that. Wiring the bot to do something useful is more involved.

Following the scope, the tech stack required three main elements:

1. A web hook that triggers certain functions when messages/commands are sent to the bot and a server to respond to those triggers.
2. A way to store the user information provided in the chat in a database.
3. A way to schedule a weekly pairings.

Another requirement for me was to do it without any cost.
Luckily I managed to find just the right online providers that have a sufficient free tier, specifically:

1. **Netlify** - hosting, as well as serverless functions and scheduled functions, that get built automatically from a connected github repo with their own environment.
2. **FaunaDB** - a graph like database with its own DSL making it easy to query and write, although my use case didn't require a very complex schema, i.e. a user table (or collection as it's called) with the user provided information and a boolean index "participants" which allows to filter the collection based on members willing to participate in the pairing. Finally, a pairs collection which stores all past pairings including date, ids, usernames and names for reference.

Once a telegram bot is created, an api key is generated allowing to communicate with the bot. A web hook was then set up between the bot and the Netlify hosted serverless function.
The bot function was written in Javascript using a JS package called `telegraf`, which wraps the Telegram API and makes it easier to register commands and send messages and retrieving the information on the specific user chatting with the bot, e.g. id and username.

For the database connection the `faunadb` package allows to query a fauna database using their specific database language, which I learned while building the bot.

One of the more non-trivial aspects of creating the bot was the question-answer part, which required waiting for user response before continuing to the next question. Luckily, telegraf has some functionality for such interaction.

# The Logic

The main logic of the bot which makes it useful is the random pairings.
The requirement of the pairing is to be exhaustive, random and not repeat a previous pairing.

Before a pairing is done, the list of participants is first randomly shuffled. I discoverd this in-place algorithm for random shuffling of an array in place, called the [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle):

```js
function shuffleArray(array) {
  // in place shuffling of an array
  for (let i = array.length - 1; i > 0; i--) {
    //  create an index between 0 and i, where i starts at the array length - 1, and gets smaller
    //  in every iteration.
    //  note that Math.random generates numbers between 0 and 1 excluding 1.
    const j = Math.floor(Math.random() * (i + 1));
    // swap the elements at indices i and j
    [array[i], array[j]] = [array[j], array[i]];
  }
}
```

The shuffling has $O(n)$ time complexity and is done in-place.

<!-- TODO: find the name of the algorithm and understand how it works -->

For N participants, there are N choose 2 ways of creating pairs, and if we consider the two orderings of each pair then we need to multiply this by 2, giving $$\frac{N!}{(N-2)!} = N\times (N-1).$$ However, since we don't want to repeat previous pairings, we cannot just pick the first random pairings. The way I implemented the pairings is to create a random shuffling, start pairing from the first half of the users in the shuffled array to the second half, if at some point a previous pair is created, restart the process and create a new shuffling. This is repeated a pre-defined maximum number of trials before the pairings is stopped due to inability to create pairs anymore or if we have a number of pairs equal to half the number of participants.

The pairing loop:

```js
let trials = 0;
const MAX_TRIALS = 5000;
let stop = false;

// paired usernames
let pairs = [];
// paired userIds
let pairs_to_store = [];

while (pairs.length != all_participants.length / 2 && stop != true) {
  pairs = [];
  pairs_to_store = [];
  trials += 1;

  shuffleArray(all_participants);
  const home = all_participants.slice(
    0,
    Math.floor(all_participants.length / 2)
  );
  const away = all_participants.slice(home.length, all_participants.length);

  // create pairs
  for (let m = 0; m < all_participants.length / 2; m += 1) {
    if (
      previous_pairs.includes(`${home[m].userId}_${away[m].userId}`) ||
      previous_pairs.includes(`${away[m].userId}_${home[m].userId}`)
    ) {
      break;
    } else {
      pairs.push([home[m], away[m]]);

      pairs_to_store.push({
        date: current_date,
        pair: `${home[m].userId}_${away[m].userId}`,
        name_1: home[m].name,
        name_2: away[m].name,
        username_1: home[m].username,
        username_2: away[m].username,
      });
    }
  }
  if (trials == MAX_TRIALS) {
    stop = true;
  }
}
```

Note that previous pairs are stored as a string with `id1_id2` and the check is if the created pair or its reverse are in the previous pairs list, which was previously retrieved from the database.

But wait, what if there are an odd number of participants? I check this before this loop, in that case I take out the last participant in the first shuffled array, and keep it separately.

```js
shuffleArray(all_participants);

// check if there are odd number of participants
// then put aside one of the participants
let last = null;
if (all_participants.length % 2 != 0) {
  last = all_participants[all_participants.length - 1];
  all_participants = all_participants.slice(0, all_participants.length - 1);
}
```

Then after the main pairing loop is finished, I add that last participant to one of the pair, such that we create a random triplet, where again no member of the triplet have been paired with one another before.

Finally, a loop is run to send messages to each member of a pair giving information about their randomly picked participant for a coffee.
This pairing function is defined to run on schedule using a cron schedule.

Finally, a few screenshots from the bot:

![bot_1](/assets/bot_1.jpeg)
![bot_2](/assets/bot_2.jpeg)
![bot_3](/assets/bot_3.jpeg)
